#!/usr/bin/env bun

/**
 * index-cli.ts
 *
 * Standalone CLI script to index Apple Notes into LanceDB.
 * Uses osascript directly with extended timeouts to avoid AppleEvent failures.
 *
 * Usage:
 *   1. Copy this file into your mcp-apple-notes-fixed directory
 *   2. Run: bun run index-cli.ts
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

const execFileAsync = promisify(execFile);

// --- Setup ---
const DATA_DIR = path.join(os.homedir(), ".mcp-apple-notes");
const DB_PATH = path.join(DATA_DIR, "data");
const TABLE_NAME = "notes";
const BATCH_SIZE = 5;

console.log("Loading embedding model (Xenova/all-MiniLM-L6-v2)...");
const extractor = await pipeline(
  "feature-extraction",
  "Xenova/all-MiniLM-L6-v2"
);
console.log("Model loaded.\n");

const td = new TurndownService();

// --- Embedding function (must match index.ts exactly) ---
@register("openai")
class OnDeviceEmbeddingFunction extends EmbeddingFunction<string> {
  toJSON(): object {
    return {};
  }
  ndims() {
    return 384;
  }
  embeddingDataType(): Float {
    return new Float32();
  }
  async computeQueryEmbeddings(data: string) {
    const output = await extractor(data, { pooling: "mean" });
    return output.data as number[];
  }
  async computeSourceEmbeddings(data: string[]) {
    return await Promise.all(
      data.map(async (item) => {
        const output = await extractor(item, { pooling: "mean" });
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

// --- AppleScript helpers using osascript with long timeout ---

/**
 * Run AppleScript via osascript with a generous 10-minute timeout.
 * This avoids the default 2-minute AppleEvent timeout in JXA.
 */
async function runAppleScript(script: string): Promise<string> {
  const { stdout } = await execFileAsync("osascript", ["-e", script], {
    timeout: 600_000, // 10 minutes
    maxBuffer: 50 * 1024 * 1024, // 50 MB
  });
  return stdout.trim();
}

/** Get the count of notes (fast, avoids fetching all properties) */
async function getNoteCount(): Promise<number> {
  const result = await runAppleScript(
    `tell application "Notes" to return count of notes`
  );
  return parseInt(result, 10);
}

/** Get all folder names */
async function getFolders(): Promise<string[]> {
  console.log("Fetching folder list from Apple Notes...");
  const result = await runAppleScript(
    `tell application "Notes" to return name of every folder`
  );
  // AppleScript returns comma-separated list
  const folders = result.split(", ").map((f) => f.trim()).filter(Boolean);
  console.log(`Found ${folders.length} folders: ${folders.join(", ")}\n`);
  return folders;
}

/** Get note count in a specific folder */
async function getNoteCountInFolder(folderName: string): Promise<number> {
  const safe = folderName.replace(/"/g, '\\"');
  const result = await runAppleScript(
    `tell application "Notes" to return count of notes of folder "${safe}"`
  );
  return parseInt(result, 10);
}

/** Get a single note's details by index within a folder (1-based) */
async function getNoteByIndex(
  folderName: string,
  index: number
): Promise<{
  title: string;
  content: string;
  creation_date: string;
  modification_date: string;
} | null> {
  const safeFolder = folderName.replace(/"/g, '\\"');
  try {
    // Fetch each property separately to keep AppleEvents small and fast
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

    return {
      title,
      content,
      creation_date: creationDate,
      modification_date: modDate,
    };
  } catch (error: any) {
    console.warn(
      `    Warning: Failed to fetch note ${index} in "${folderName}": ${error.message}`
    );
    return null;
  }
}

// --- Main ---
async function main() {
  const startTime = Date.now();

  // Ensure data directory exists
  await fs.mkdir(DATA_DIR, { recursive: true });

  // Quick check: how many notes total?
  const totalCount = await getNoteCount();
  console.log(`Total notes in Apple Notes: ${totalCount}\n`);

  // Connect to LanceDB
  console.log(`Connecting to LanceDB at ${DB_PATH}...`);
  const db = await lancedb.connect(DB_PATH);

  // Drop existing table if present
  try {
    await db.dropTable(TABLE_NAME);
    console.log("Dropped existing notes table.");
  } catch {
    // Table didn't exist
  }

  // Create table with the embedding schema
  const notesTable = await db.createEmptyTable(TABLE_NAME, notesTableSchema);
  console.log("Created fresh notes table with embedding schema.\n");

  // Step 1: Get all folders
  const folders = await getFolders();
  if (folders.length === 0) {
    console.log("No folders found. Exiting.");
    return;
  }

  let totalProcessed = 0;
  let totalErrors = 0;

  // Step 2: Process each folder, note by note
  for (const folder of folders) {
    console.log(`--- Folder: "${folder}" ---`);

    let noteCount: number;
    try {
      noteCount = await getNoteCountInFolder(folder);
    } catch (error: any) {
      console.warn(`  Skipping folder "${folder}": ${error.message}`);
      continue;
    }

    console.log(`  ${noteCount} notes`);
    if (noteCount === 0) continue;

    // Process in batches of BATCH_SIZE
    for (let i = 1; i <= noteCount; i += BATCH_SIZE) {
      const end = Math.min(i + BATCH_SIZE - 1, noteCount);
      const batchNum = Math.ceil(i / BATCH_SIZE);
      const totalBatches = Math.ceil(noteCount / BATCH_SIZE);

      console.log(
        `  Batch ${batchNum}/${totalBatches} (notes ${i}-${end})`
      );

      const chunks: {
        title: string;
        content: string;
        creation_date: string;
        modification_date: string;
      }[] = [];

      for (let idx = i; idx <= end; idx++) {
        const detail = await getNoteByIndex(folder, idx);
        if (!detail) {
          totalErrors++;
          continue;
        }

        console.log(`    [${idx}/${noteCount}] "${detail.title}"`);

        // Convert HTML to markdown
        let content: string;
        try {
          content = detail.content ? td.turndown(detail.content) : "";
        } catch {
          content = detail.content || "";
        }

        chunks.push({
          title: detail.title,
          content,
          creation_date: detail.creation_date,
          modification_date: detail.modification_date,
        });
      }

      if (chunks.length > 0) {
        try {
          await notesTable.add(chunks);
          totalProcessed += chunks.length;
          console.log(
            `    Added ${chunks.length} notes. Running total: ${totalProcessed}`
          );
        } catch (error: any) {
          console.error(`    DB error: ${error.message}`);
          totalErrors += chunks.length;
        }
      }

      // Small delay between batches
      await new Promise((resolve) => setTimeout(resolve, 300));
    }

    console.log("");
  }

  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
  console.log(`========================================`);
  console.log(`Indexing complete!`);
  console.log(`  Notes indexed: ${totalProcessed}`);
  console.log(`  Errors: ${totalErrors}`);
  console.log(`  Time: ${elapsed}s`);
  console.log(`  DB path: ${DB_PATH}`);
  console.log(`========================================`);
  console.log(
    `\nYour notes are now searchable via Claude. Restart the MCP server if it's running.`
  );
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
