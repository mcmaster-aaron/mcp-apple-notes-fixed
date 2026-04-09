#!/usr/bin/env bun

/**
 * index-cli.ts
 *
 * Standalone CLI script to index Apple Notes into LanceDB.
 * Uses osascript directly with extended timeouts to avoid AppleEvent failures.
 * Uses EmbeddingGemma-300M for on-device embeddings (768-dim).
 * Supports content chunking for long notes.
 *
 * Usage:
 *   Full rebuild (default):  bun run index-cli.ts
 *   Incremental update:      bun run index-cli.ts --incremental
 */

import * as lancedb from "@lancedb/lancedb";
import {
  EmbeddingFunction,
  LanceSchema,
  register,
} from "@lancedb/lancedb/embedding";
import { type Float, Float32, Utf8 } from "apache-arrow";
import { pipeline } from "@huggingface/transformers";
import TurndownService from "turndown";
import path from "node:path";
import os from "node:os";
import fs from "node:fs/promises";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

// =============================================================================
// Constants
// =============================================================================

const DATA_DIR = path.join(os.homedir(), ".mcp-apple-notes");
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
  console.log(`[${timestamp}] [${level}] ${message}`);
}

// =============================================================================
// Parse CLI args (Change G)
// =============================================================================

const isIncremental = process.argv.includes("--incremental");

// =============================================================================
// Setup
// =============================================================================

log("INFO", `Loading embedding model: ${MODEL_ID}...`);
const extractor = await pipeline("feature-extraction", MODEL_ID);
log("INFO", "Embedding model loaded.");

const td = new TurndownService();
const execFileAsync = promisify(execFile);

const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: CHUNK_SIZE,
  chunkOverlap: CHUNK_OVERLAP,
  separators: ["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " ", ""],
});

// =============================================================================
// Embedding function (Change A - must match index.ts exactly)
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
    const results: number[][] = [];
    for (const item of data) {
      const output = await extractor(item, { pooling: "mean", normalize: true });
      results.push(output.data as number[]);
    }
    return results;
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
// AppleScript utilities (Changes E and I/F)
// =============================================================================

/** Escape a string for safe interpolation into an AppleScript double-quoted string. */
function escapeForAppleScript(str: string): string {
  return str
    .replace(/\\/g, "\\\\")
    .replace(/"/g, '\\"')
    .replace(/\u201C/g, '\\"')
    .replace(/\u201D/g, '\\"')
    .replace(/\r/g, "");
}

async function runAppleScript(script: string): Promise<string> {
  const { stdout } = await execFileAsync("osascript", ["-e", script], {
    timeout: APPLESCRIPT_TIMEOUT,
    maxBuffer: APPLESCRIPT_MAX_BUFFER,
  });
  return stdout.trim();
}

async function getFolders(): Promise<string[]> {
  log("INFO", "Fetching folder list from Apple Notes...");
  const result = await runAppleScript(
    `tell application "Notes" to return name of every folder`
  );
  const folders = result.split(", ").map((f) => f.trim()).filter(Boolean);
  log("INFO", `Found ${folders.length} folders.`);
  return folders;
}

async function getNoteCountInFolder(folderName: string): Promise<number> {
  const safe = escapeForAppleScript(folderName);
  const result = await runAppleScript(
    `tell application "Notes" to return count of notes of folder "${safe}"`
  );
  return parseInt(result, 10);
}

async function getNoteByIndex(
  folderName: string,
  index: number
): Promise<{
  title: string;
  content: string;
  creation_date: string;
  modification_date: string;
} | null> {
  const safeFolder = escapeForAppleScript(folderName);
  try {
    const title = await runAppleScript(
      `tell application "Notes" to return name of note ${index} of folder "${safeFolder}"`
    );
    const content = await runAppleScript(
      `tell application "Notes" to return body of note ${index} of folder "${safeFolder}"`
    );
    const creationDate = await runAppleScript(
      `tell application "Notes" to return creation date of note ${index} of folder "${safeFolder}" as string`
    );
    const modDate = await runAppleScript(
      `tell application "Notes" to return modification date of note ${index} of folder "${safeFolder}" as string`
    );
    if (!title) return null;
    return { title, content, creation_date: creationDate, modification_date: modDate };
  } catch (error: any) {
    log("WARN", `Failed to fetch note ${index} in "${folderName}": ${error.message}`);
    return null;
  }
}

async function getNoteMetaByIndex(
  folderName: string,
  index: number
): Promise<{ title: string; modification_date: string } | null> {
  const safeFolder = escapeForAppleScript(folderName);
  try {
    const title = await runAppleScript(
      `tell application "Notes" to return name of note ${index} of folder "${safeFolder}"`
    );
    const modDate = await runAppleScript(
      `tell application "Notes" to return modification date of note ${index} of folder "${safeFolder}" as string`
    );
    if (!title) return null;
    return { title, modification_date: modDate };
  } catch (error: any) {
    log("WARN", `Failed to fetch metadata for note ${index} in "${folderName}": ${error.message}`);
    return null;
  }
}

async function getNoteDetailsByTitle(title: string): Promise<{
  title: string;
  content: string;
  creation_date: string;
  modification_date: string;
} | null> {
  const safeTitle = escapeForAppleScript(title);
  try {
    const noteTitle = await runAppleScript(
      `tell application "Notes"\nset theNote to first note whose name is "${safeTitle}"\nreturn name of theNote\nend tell`
    );
    const content = await runAppleScript(
      `tell application "Notes"\nset theNote to first note whose name is "${safeTitle}"\nreturn body of theNote\nend tell`
    );
    const creationDate = await runAppleScript(
      `tell application "Notes"\nset theNote to first note whose name is "${safeTitle}"\nreturn creation date of theNote as string\nend tell`
    );
    const modDate = await runAppleScript(
      `tell application "Notes"\nset theNote to first note whose name is "${safeTitle}"\nreturn modification date of theNote as string\nend tell`
    );
    if (!noteTitle) return null;
    return { title: noteTitle, content, creation_date: creationDate, modification_date: modDate };
  } catch (error: any) {
    log("WARN", `Failed to fetch note by title "${title}": ${error.message}`);
    return null;
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
// Full rebuild
// =============================================================================

async function fullRebuild(): Promise<void> {
  const startTime = Date.now();

  await fs.mkdir(DATA_DIR, { recursive: true });

  log("INFO", `Connecting to LanceDB at ${DB_PATH}...`);
  const db = await lancedb.connect(DB_PATH);

  try {
    await db.dropTable(TABLE_NAME);
    log("INFO", "Dropped existing notes table.");
  } catch {
    // Table didn't exist
  }

  const notesTable = await db.createEmptyTable(TABLE_NAME, notesTableSchema);
  log("INFO", "Created fresh notes table with embedding schema.");

  const folders = await getFolders();
  if (folders.length === 0) {
    log("INFO", "No folders found. Exiting.");
    return;
  }

  let totalProcessed = 0;
  let totalChunks = 0;
  let totalErrors = 0;

  for (const folder of folders) {
    log("INFO", `--- Folder: "${folder}" ---`);

    let noteCount: number;
    try {
      noteCount = await getNoteCountInFolder(folder);
    } catch (error: any) {
      log("WARN", `Skipping folder "${folder}": ${error.message}`);
      continue;
    }

    log("INFO", `  ${noteCount} notes`);
    if (noteCount === 0) continue;

    for (let i = 1; i <= noteCount; i += BATCH_SIZE) {
      const end = Math.min(i + BATCH_SIZE - 1, noteCount);
      const batchNum = Math.ceil(i / BATCH_SIZE);
      const totalBatches = Math.ceil(noteCount / BATCH_SIZE);

      log("INFO", `  Batch ${batchNum}/${totalBatches} (notes ${i}-${end})`);

      const allChunks: NoteChunk[] = [];

      for (let idx = i; idx <= end; idx++) {
        const detail = await getNoteByIndex(folder, idx);
        if (!detail) {
          totalErrors++;
          continue;
        }

        log("INFO", `    [${idx}/${noteCount}] "${detail.title}"`);

        try {
          const chunks = await chunkNote(detail);
          allChunks.push(...chunks);
          log("INFO", `      -> ${chunks.length} chunk(s)`);
        } catch (error: any) {
          log("ERROR", `      Chunking failed: ${error.message}`);
          totalErrors++;
        }
      }

      if (allChunks.length > 0) {
        try {
          await notesTable.add(allChunks);
          totalProcessed += Math.min(end - i + 1, BATCH_SIZE);
          totalChunks += allChunks.length;
          log("INFO", `    Added ${allChunks.length} chunks. Running total: ${totalChunks}`);
        } catch (error: any) {
          log("ERROR", `    DB error: ${error.message}`);
          totalErrors++;
        }
      }

      await new Promise((resolve) => setTimeout(resolve, 300));
    }
  }

  // Build FTS index
  try {
    await notesTable.createIndex("content", {
      config: lancedb.Index.fts(),
      replace: true,
    });
    log("INFO", "FTS index created.");
  } catch (error: any) {
    log("WARN", `Failed to create FTS index: ${error.message}`);
  }

  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
  log("INFO", "========================================");
  log("INFO", "Indexing complete!");
  log("INFO", `  Notes indexed: ${totalProcessed}`);
  log("INFO", `  Total chunks: ${totalChunks}`);
  log("INFO", `  Errors: ${totalErrors}`);
  log("INFO", `  Time: ${elapsed}s`);
  log("INFO", `  DB path: ${DB_PATH}`);
  log("INFO", "========================================");
  log("INFO", "Your notes are now searchable via Claude. Restart the MCP server if it's running.");
}

// =============================================================================
// Incremental update (Change G)
// =============================================================================

async function incrementalUpdate(): Promise<void> {
  const startTime = Date.now();

  await fs.mkdir(DATA_DIR, { recursive: true });

  log("INFO", `Connecting to LanceDB at ${DB_PATH}...`);
  const db = await lancedb.connect(DB_PATH);

  const notesTable = await db.createEmptyTable(TABLE_NAME, notesTableSchema, {
    mode: "create",
    existOk: true,
  });

  log("INFO", "Starting incremental index update...");

  // Fetch current state from Apple Notes
  const appleNotesMeta = new Map<string, string>();
  const folders = await getFolders();

  for (const folder of folders) {
    let noteCount: number;
    try {
      noteCount = await getNoteCountInFolder(folder);
    } catch (error: any) {
      log("WARN", `Skipping folder "${folder}": ${error.message}`);
      continue;
    }
    if (noteCount === 0) continue;

    for (let i = 1; i <= noteCount; i++) {
      const meta = await getNoteMetaByIndex(folder, i);
      if (meta && meta.title) {
        appleNotesMeta.set(meta.title, meta.modification_date);
      }
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
  }

  // Fetch current index state
  const indexedMeta = new Map<string, string>();
  try {
    const rows = await notesTable
      .search("")
      .select(["title", "modification_date"])
      .limit(100000)
      .toArray();
    for (const row of rows) {
      if (row.title && !indexedMeta.has(row.title)) {
        indexedMeta.set(row.title, row.modification_date);
      }
    }
  } catch (error: any) {
    log("WARN", `Could not read indexed metadata: ${error.message}`);
  }

  log("INFO", `Apple Notes: ${appleNotesMeta.size} notes. Index: ${indexedMeta.size} titles.`);

  // Classify changes
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
    log("INFO", "Index is up to date. No changes needed.");
    return;
  }

  let errors = 0;

  // Delete chunks for deleted and modified notes
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
        log("INFO", `Inserted ${allChunks.length} chunks.`);
      } catch (error: any) {
        log("ERROR", `Failed to insert batch: ${error.message}`);
        errors += allChunks.length;
      }
    }

    await new Promise((resolve) => setTimeout(resolve, 300));
  }

  // Rebuild FTS index
  try {
    await notesTable.createIndex("content", {
      config: lancedb.Index.fts(),
      replace: true,
    });
    log("INFO", "FTS index rebuilt.");
  } catch (error: any) {
    log("WARN", `Failed to rebuild FTS index: ${error.message}`);
  }

  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
  log("INFO", `Incremental update complete in ${elapsed}s. Added: ${newTitles.length}, Modified: ${modifiedTitles.length}, Deleted: ${deletedTitles.length}, Errors: ${errors}.`);
}

// =============================================================================
// Main
// =============================================================================

async function main() {
  if (isIncremental) {
    log("INFO", "Running incremental update...");
    await incrementalUpdate();
  } else {
    log("INFO", "Running full rebuild...");
    await fullRebuild();
  }
}

main().catch((err) => {
  log("ERROR", `Fatal error: ${err.message}`);
  process.exit(1);
});
