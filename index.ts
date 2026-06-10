import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { z } from "zod";
import * as lancedb from "@lancedb/lancedb";
import path from "node:path";
import os from "node:os";
import TurndownService from "turndown";
import fs from "node:fs/promises";
import {
  EmbeddingFunction,
  LanceSchema,
  register,
} from "@lancedb/lancedb/embedding";
import { type Float, Float32, Utf8 } from "apache-arrow";
import { pipeline } from "@huggingface/transformers";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

// =============================================================================
// Constants
// =============================================================================

const DATA_DIR = path.join(os.homedir(), ".mcp-apple-notes");
const HTTP_PORT = 7891;
const DB_PATH = path.join(DATA_DIR, "data");
const TABLE_NAME = "notes";
const MODEL_ID = "Xenova/bge-small-en-v1.5";
const EMBEDDING_DIMS = 384;
const CHUNK_SIZE = 1500;
const CHUNK_OVERLAP = 200;
const BATCH_SIZE = 5;
const APPLESCRIPT_TIMEOUT = 600_000; // 10 minutes
const APPLESCRIPT_MAX_BUFFER = 50 * 1024 * 1024; // 50 MB

// =============================================================================
// Logging (Change L)
// =============================================================================

type LogLevel = "INFO" | "WARN" | "ERROR";

function log(level: LogLevel, message: string): void {
  const timestamp = new Date().toISOString();
  process.stderr.write(`[${timestamp}] [${level}] ${message}\n`);
}

// =============================================================================
// Ensure data directory exists
// =============================================================================

await fs.mkdir(DATA_DIR, { recursive: true });

// =============================================================================
// Model and DB setup (Change A)
// =============================================================================

log("INFO", `Loading embedding model: ${MODEL_ID}...`);
const extractor = await pipeline("feature-extraction", MODEL_ID);
log("INFO", "Embedding model loaded.");

const td = new TurndownService();
const db = await lancedb.connect(DB_PATH);

// =============================================================================
// Text splitter for chunking (Change J)
// =============================================================================

const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: CHUNK_SIZE,
  chunkOverlap: CHUNK_OVERLAP,
  separators: ["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " ", ""],
});

// =============================================================================
// Embedding function (Change A)
// =============================================================================

@register("openai")
class OnDeviceEmbeddingFunction extends EmbeddingFunction<string> {
  toJSON(): object {
    return {};
  }
  ndims() {
    return EMBEDDING_DIMS;
  }
  embeddingDataType(): Float {
    return new Float32();
  }
  async computeQueryEmbeddings(data: string) {
    const output = await extractor(data, { pooling: "mean", normalize: true });
    return output.data as number[];
  }
  async computeSourceEmbeddings(data: string[]) {
    return Promise.all(
      data.map(async (item) => {
        const output = await extractor(item, { pooling: "mean", normalize: true });
        return output.data as number[];
      })
    );
  }
}

const func = new OnDeviceEmbeddingFunction();

const notesTableSchema = LanceSchema({
  title: func.sourceField(new Utf8()),
  content: func.sourceField(new Utf8()),
  creation_date: func.sourceField(new Utf8()),
  modification_date: func.sourceField(new Utf8()),
  vector: func.vectorField(),
});

// =============================================================================
// AppleScript utilities (Changes E and I)
// =============================================================================

const execFileAsync = promisify(execFile);

/**
 * Escape a string for safe interpolation into an AppleScript double-quoted string.
 * Handles backslashes, double quotes, and curly/smart quotes. (Change I)
 */
function escapeForAppleScript(str: string): string {
  return str
    .replace(/\\/g, "\\\\")
    .replace(/"/g, '\\"')
    .replace(/\u201C/g, '\\"')
    .replace(/\u201D/g, '\\"')
    .replace(/\r/g, "");
}

/** Run AppleScript via osascript with a generous timeout. (Change E) */
async function runAppleScript(script: string): Promise<string> {
  const { stdout } = await execFileAsync("osascript", ["-e", script], {
    timeout: APPLESCRIPT_TIMEOUT,
    maxBuffer: APPLESCRIPT_MAX_BUFFER,
  });
  return stdout.trim();
}

/** Get a note's full details by title in a single AppleScript call */
async function getNoteDetailsByTitle(title: string): Promise<{
  title: string;
  content: string;
  creation_date: string;
  modification_date: string;
} | null> {
  const safeTitle = escapeForAppleScript(title);
  try {
    // Fetch name, creation_date, mod_date together; body separately to avoid
    // delimiter collisions with arbitrary HTML content.
    const meta = await runAppleScript(
      `tell application "Notes"\n` +
      `set n to first note whose name is "${safeTitle}"\n` +
      `return (name of n) & "|||" & (creation date of n as string) & "|||" & (modification date of n as string)\n` +
      `end tell`
    );
    if (!meta) return null;
    const [noteTitle, creationDate, modDate] = meta.split("|||");
    const content = await runAppleScript(
      `tell application "Notes"\nreturn body of (first note whose name is "${safeTitle}")\nend tell`
    );
    return { title: noteTitle.trim(), content, creation_date: creationDate.trim(), modification_date: modDate.trim() };
  } catch (error: any) {
    log("WARN", `Failed to fetch note by title "${title}": ${error.message}`);
    return null;
  }
}

/** Create a new note in Apple Notes */
async function createNote(title: string, content: string): Promise<boolean> {
  const safeTitle = escapeForAppleScript(title);
  const safeContent = escapeForAppleScript(content).replace(/\n/g, "\\n");
  try {
    await runAppleScript(
      `tell application "Notes"\nmake new note with properties {name:"${safeTitle}", body:"${safeContent}"}\nend tell`
    );
    return true;
  } catch (error: any) {
    log("ERROR", `Failed to create note "${title}": ${error.message}`);
    return false;
  }
}

// =============================================================================
// Chunking helper (Change J)
// =============================================================================

interface NoteChunk extends Record<string, unknown> {
  title: string;
  content: string;
  creation_date: string;
  modification_date: string;
}

/**
 * Convert a note's HTML content to Markdown and split into chunks.
 * Each chunk shares the same title/creation_date/modification_date as the parent note.
 */
async function chunkNote(note: {
  title: string;
  content: string;
  creation_date: string;
  modification_date: string;
}): Promise<NoteChunk[]> {
  let markdown: string;
  try {
    markdown = note.content ? td.turndown(note.content) : "";
  } catch {
    markdown = note.content || "";
  }

  if (!markdown || markdown.trim().length === 0) {
    return [{
      title: note.title || "Untitled",
      content: "",
      creation_date: note.creation_date || new Date().toISOString(),
      modification_date: note.modification_date || new Date().toISOString(),
    }];
  }

  const chunks = await textSplitter.splitText(markdown);
  return chunks.map((chunk) => ({
    title: note.title || "Untitled",
    content: chunk,
    creation_date: note.creation_date || new Date().toISOString(),
    modification_date: note.modification_date || new Date().toISOString(),
  }));
}

// =============================================================================
// Table management (Change K)
// =============================================================================

async function createNotesTable(overrideName?: string): Promise<{ notesTable: lancedb.Table; time: number }> {
  const start = performance.now();
  const notesTable = await db.createEmptyTable(
    overrideName || TABLE_NAME,
    notesTableSchema,
    { mode: "create", existOk: true }
  );
  await ensureFtsIndex(notesTable);
  return { notesTable, time: performance.now() - start };
}

async function ensureFtsIndex(notesTable: lancedb.Table): Promise<void> {
  try {
    const indices = await notesTable.listIndices();
    if (!indices.find((index) => index.name === "content_idx")) {
      await notesTable.createIndex("content", {
        config: lancedb.Index.fts(),
        replace: true,
      });
    }
  } catch (error: any) {
    log("WARN", `Failed to ensure FTS index: ${error.message}`);
  }
}

async function rebuildFtsIndex(notesTable: lancedb.Table): Promise<void> {
  try {
    await notesTable.createIndex("content", {
      config: lancedb.Index.fts(),
      replace: true,
    });
    log("INFO", "FTS index rebuilt.");
  } catch (error: any) {
    log("WARN", `Failed to rebuild FTS index: ${error.message}`);
  }
}

// =============================================================================
// Incremental indexing (Changes B, C, D, J, K)
// =============================================================================

/** Fetch all note metadata from Apple Notes in a single AppleScript call */
async function fetchAllNoteMeta(): Promise<Map<string, string>> {
  const notesMeta = new Map<string, string>();
  log("INFO", "Fetching all note metadata in a single AppleScript call...");

  const result = await runAppleScript(
    `tell application "Notes"\n` +
    `set output to ""\n` +
    `repeat with f in (every folder)\n` +
    `if (count of notes of f) > 0 then\n` +
    `repeat with n in (every note of f)\n` +
    `set output to output & (name of n) & "|||" & ((modification date of n) as string) & "~~~"\n` +
    `end repeat\n` +
    `end if\n` +
    `end repeat\n` +
    `return output\n` +
    `end tell`
  );

  if (result) {
    for (const entry of result.split("~~~")) {
      const trimmed = entry.trim();
      if (!trimmed) continue;
      const sepIdx = trimmed.indexOf("|||");
      if (sepIdx !== -1) {
        const title = trimmed.substring(0, sepIdx).trim();
        const modDate = trimmed.substring(sepIdx + 3).trim();
        if (title) notesMeta.set(title, modDate);
      }
    }
  }

  log("INFO", `Found ${notesMeta.size} notes.`);
  return notesMeta;
}

/** Get indexed notes metadata from LanceDB (groups by title) */
async function getIndexedNoteMeta(notesTable: lancedb.Table): Promise<Map<string, string>> {
  const indexed = new Map<string, string>();
  try {
    const rows = await notesTable
      .query()
      .select(["title", "modification_date"])
      .limit(100000)
      .toArray();
    for (const row of rows) {
      if (row.title && !indexed.has(row.title)) {
        indexed.set(row.title, row.modification_date);
      }
    }
  } catch (error: any) {
    log("WARN", `Could not read indexed metadata: ${error.message}`);
  }
  return indexed;
}

/** Run incremental indexing: only process new, modified, and deleted notes */
async function incrementalIndexNotes(notesTable: lancedb.Table): Promise<{
  added: number;
  modified: number;
  deleted: number;
  errors: number;
}> {
  const startTime = Date.now();
  log("INFO", "Starting incremental index update...");

  const [appleNotesMeta, indexedMeta] = await Promise.all([
    fetchAllNoteMeta(),
    getIndexedNoteMeta(notesTable),
  ]);

  log("INFO", `Apple Notes: ${appleNotesMeta.size} notes. Index: ${indexedMeta.size} titles.`);

  const newTitles: string[] = [];
  const modifiedTitles: string[] = [];
  const deletedTitles: string[] = [];

  for (const [title, modDate] of appleNotesMeta) {
    if (!indexedMeta.has(title)) {
      newTitles.push(title);
    } else if (indexedMeta.get(title) !== modDate) {
      modifiedTitles.push(title);
    }
  }

  for (const title of indexedMeta.keys()) {
    if (!appleNotesMeta.has(title)) {
      deletedTitles.push(title);
    }
  }

  log("INFO", `Changes: ${newTitles.length} new, ${modifiedTitles.length} modified, ${deletedTitles.length} deleted.`);

  if (newTitles.length === 0 && modifiedTitles.length === 0 && deletedTitles.length === 0) {
    log("INFO", "Index is up to date.");
    return { added: 0, modified: 0, deleted: 0, errors: 0 };
  }

  let errors = 0;

  // Delete all chunks for deleted and modified notes
  const titlesToRemove = [...deletedTitles, ...modifiedTitles];
  for (const title of titlesToRemove) {
    try {
      const safeTitle = title.replace(/'/g, "''");
      await notesTable.delete(`title = '${safeTitle}'`);
      log("INFO", `Removed chunks for: "${title}"`);
    } catch (error: any) {
      log("ERROR", `Failed to delete chunks for "${title}": ${error.message}`);
      errors++;
    }
  }

  // Fetch, chunk, and insert new and modified notes
  const titlesToAdd = [...newTitles, ...modifiedTitles];
  for (let i = 0; i < titlesToAdd.length; i += BATCH_SIZE) {
    const batch = titlesToAdd.slice(i, i + BATCH_SIZE);
    const allChunks: NoteChunk[] = [];

    for (const title of batch) {
      try {
        const note = await getNoteDetailsByTitle(title);
        if (!note) {
          log("WARN", `Could not fetch note "${title}" for indexing.`);
          errors++;
          continue;
        }
        const chunks = await chunkNote(note);
        allChunks.push(...chunks);
        log("INFO", `Chunked "${title}" into ${chunks.length} chunk(s).`);
      } catch (error: any) {
        log("ERROR", `Error processing "${title}": ${error.message}`);
        errors++;
      }
    }

    if (allChunks.length > 0) {
      try {
        await notesTable.add(allChunks);
        log("INFO", `Inserted ${allChunks.length} chunks for batch ${Math.floor(i / BATCH_SIZE) + 1}.`);
      } catch (error: any) {
        log("ERROR", `Failed to insert batch: ${error.message}`);
        errors += allChunks.length;
      }
    }

    await new Promise((resolve) => setTimeout(resolve, 300));
  }

  await rebuildFtsIndex(notesTable);

  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
  log("INFO", `Incremental update complete in ${elapsed}s. Added: ${newTitles.length}, Modified: ${modifiedTitles.length}, Deleted: ${deletedTitles.length}, Errors: ${errors}.`);

  return { added: newTitles.length, modified: modifiedTitles.length, deleted: deletedTitles.length, errors };
}

// =============================================================================
// Zod schemas
// =============================================================================

const QueryNotesSchema = z.object({ query: z.string() });
const GetNoteSchema = z.object({ title: z.string() });
const CreateNoteSchema = z.object({ title: z.string(), content: z.string() });

// =============================================================================
// MCP Server
// =============================================================================

const server = new Server(
  { name: "my-apple-notes-mcp", version: "2.0.0" },
  { capabilities: { tools: {} } }
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: "list-notes",
      description: "Lists just the titles of all my Apple Notes",
      inputSchema: { type: "object", properties: {}, required: [] },
    },
    {
      name: "index-notes",
      description: "Incrementally update the Apple Notes search index. Only re-indexes new, modified, and deleted notes. Runs automatically on server startup, but can be triggered manually.",
      inputSchema: { type: "object", properties: {}, required: [] },
    },
    {
      name: "get-note",
      description: "Get a note full content and details by title",
      inputSchema: { type: "object", properties: { title: { type: "string" } }, required: ["title"] },
    },
    {
      name: "search-notes",
      description: "Search for notes by title or content. Returns ranked titles with relevance scores. Use get-note to fetch full content of specific results.",
      inputSchema: { type: "object", properties: { query: { type: "string" } }, required: ["query"] },
    },
    {
      name: "create-note",
      description: "Create a new Apple Note with specified title and content. Must be in HTML format WITHOUT newlines",
      inputSchema: { type: "object", properties: { title: { type: "string" }, content: { type: "string" } }, required: ["title", "content"] },
    },
  ],
}));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { notesTable } = await createNotesTable();
  const { name, arguments: args } = request.params;

  try {
    if (name === "create-note") {
      const { title, content } = CreateNoteSchema.parse(args);
      const success = await createNote(title, content);
      return createTextResponse(success ? `Created note "${title}" successfully.` : `Failed to create note "${title}".`);
    } else if (name === "list-notes") {
      const count = await notesTable.countRows();
      return createTextResponse(`There are ${count} note chunks in your Apple Notes search index.`);
    } else if (name === "get-note") {
      const { title } = GetNoteSchema.parse(args);
      const note = await getNoteDetailsByTitle(title);
      return createTextResponse(JSON.stringify(note));
    } else if (name === "index-notes") {
      const result = await incrementalIndexNotes(notesTable);
      return createTextResponse(`Incremental index update complete. Added: ${result.added}, Modified: ${result.modified}, Deleted: ${result.deleted}, Errors: ${result.errors}.`);
    } else if (name === "search-notes") {
      const { query } = QueryNotesSchema.parse(args);
      const results = await searchAndCombineResults(notesTable, query);
      return createTextResponse(JSON.stringify(results));
    } else {
      throw new Error(`Unknown tool: ${name}`);
    }
  } catch (error: any) {
    if (error instanceof z.ZodError) {
      throw new Error(`Invalid arguments: ${error.errors.map((e) => `${e.path.join(".")}: ${e.message}`).join(", ")}`);
    }
    throw error;
  }
});

// =============================================================================
// Search
// =============================================================================

const createTextResponse = (text: string) => ({
  content: [{ type: "text" as const, text }],
});

export const searchAndCombineResults = async (
  notesTable: lancedb.Table,
  query: string,
  limit = 10
) => {
  const [vectorResults, ftsSearchResults] = await Promise.all([
    notesTable.search(query, "vector").limit(limit * 3).toArray(),
    notesTable.search(query, "fts", "content").limit(limit * 3).toArray(),
  ]);

  const k = 60;
  const scores = new Map<string, { score: number; title: string }>();

  const processResults = (results: any[], startRank: number) => {
    results.forEach((result, idx) => {
      const title = result.title;
      const score = 1 / (k + startRank + idx);
      const existing = scores.get(title);
      if (existing) {
        existing.score += score;
      } else {
        scores.set(title, { score, title });
      }
    });
  };

  processResults(vectorResults, 0);
  processResults(ftsSearchResults, 0);

  return Array.from(scores.values())
    .sort((a, b) => b.score - a.score)
    .slice(0, limit)
    .map(({ title, score }) => ({
      title,
      relevance: Math.round(score * 10000) / 10000,
    }));
};

// =============================================================================
// HTTP proxy — Notes writes from processes without direct Apple Events access
// =============================================================================

async function updateNote(folder: string, title: string, content: string): Promise<string> {
  const sf = escapeForAppleScript(folder);
  const st = escapeForAppleScript(title);
  const sc = escapeForAppleScript(content).replace(/\n/g, "\\n");

  // Use Notes' built-in predicate search — much faster than manual loops on large databases.
  // Iterate matches to skip any that are in Recently Deleted.
  const findResult = await runAppleScript(
    `tell application "Notes"\n` +
    `set matchingNotes to (every note whose name is "${st}")\n` +
    `repeat with n in matchingNotes\n` +
    `try\n` +
    `if name of folder of n is not "Recently Deleted" then\n` +
    `set body of n to "${sc}"\n` +
    `return "updated"\n` +
    `end if\n` +
    `end try\n` +
    `end repeat\n` +
    `return "not-found"\n` +
    `end tell`
  );

  if (findResult === "updated") return "updated";

  // Note does not exist — find the target folder and create it.
  return await runAppleScript(
    `tell application "Notes"\n` +
    `set targetFolder to missing value\n` +
    `repeat with theAccount in accounts\n` +
    `repeat with theFolder in folders of theAccount\n` +
    `if name of theFolder is "${sf}" then\n` +
    `set targetFolder to theFolder\n` +
    `exit repeat\n` +
    `end if\n` +
    `repeat with subFolder in folders of theFolder\n` +
    `if name of subFolder is "${sf}" then\n` +
    `set targetFolder to subFolder\n` +
    `exit repeat\n` +
    `end if\n` +
    `end repeat\n` +
    `if targetFolder is not missing value then exit repeat\n` +
    `end repeat\n` +
    `if targetFolder is not missing value then exit repeat\n` +
    `end repeat\n` +
    `if targetFolder is not missing value then\n` +
    `tell targetFolder\n` +
    `make new note with properties {name:"${st}", body:"${sc}"}\n` +
    `end tell\n` +
    `return "created-in-folder"\n` +
    `else\n` +
    `make new note with properties {name:"${st}", body:"${sc}"}\n` +
    `return "created-default"\n` +
    `end if\n` +
    `end tell`
  );
}

const MAX_REQUEST_BODY = 1_048_576; // 1 MB

function readRequestBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let totalSize = 0;
    req.on("data", (chunk: Buffer) => {
      totalSize += chunk.length;
      if (totalSize > MAX_REQUEST_BODY) {
        reject(new Error("request body too large"));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
}

const httpServer = createServer(async (req: IncomingMessage, res: ServerResponse) => {
  const url = new URL(req.url ?? "/", `http://127.0.0.1:${HTTP_PORT}`);

  if (req.method === "GET" && url.pathname === "/health") {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ status: "ok" }));
    return;
  }

  if (req.method === "POST" && url.pathname === "/update-note") {
    try {
      const body = await readRequestBody(req);
      const { folder, title, content } = JSON.parse(body) as { folder: string; title: string; content: string };
      if (!folder || !title || !content) {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "folder, title, and content are required" }));
        return;
      }
      const result = await updateNote(folder, title, content);
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ result }));
    } catch (err: any) {
      log("ERROR", `HTTP /update-note: ${err.message}`);
      res.writeHead(500, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "internal error" }));
    }
    return;
  }

  res.writeHead(404, { "Content-Type": "application/json" });
  res.end(JSON.stringify({ error: "not found" }));
});

httpServer.listen(HTTP_PORT, "127.0.0.1", () => {
  log("INFO", `HTTP proxy listening on http://127.0.0.1:${HTTP_PORT}`);
});

// =============================================================================
// Start server and run startup indexing (Changes C and M)
// =============================================================================

const transport = new StdioServerTransport();
await server.connect(transport);
log("INFO", "MCP Apple Notes server v2.0.0 running on stdio.");

// Auto-index on startup with graceful failure handling
(async () => {
  try {
    const { notesTable } = await createNotesTable();
    await incrementalIndexNotes(notesTable);
  } catch (error: any) {
    log("WARN", `Startup indexing failed (Apple Notes may be unavailable): ${error.message}`);
    log("WARN", "Server is running. Use the index-notes tool to retry later.");
  }
})();
