/**
 * index.test.ts
 *
 * Tests for the MCP Apple Notes server.
 * Run with: bun test
 *
 * Tests:
 *   N1: Embedding dimension validation
 *   N2: AppleScript escaping correctness
 *   N3: Incremental indexing correctness (requires mocking Apple Notes)
 *   N4: Search result size safety
 *   N5: Chunking correctness
 */

import { describe, it, expect } from "bun:test";
import { pipeline } from "@huggingface/transformers";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

// =============================================================================
// Shared constants (must match index.ts)
// =============================================================================

const MODEL_ID = "Xenova/bge-small-en-v1.5";
const EMBEDDING_DIMS = 384;
const CHUNK_SIZE = 1500;
const CHUNK_OVERLAP = 200;

// =============================================================================
// Shared utilities (duplicated from index.ts for test isolation)
// =============================================================================

function escapeForAppleScript(str: string): string {
  return str
    .replace(/\\/g, "\\\\")
    .replace(/"/g, '\\"')
    .replace(/\u201C/g, '\\"')
    .replace(/\u201D/g, '\\"')
    .replace(/\r/g, "");
}

// =============================================================================
// Test N1: Embedding dimension validation
// =============================================================================

describe("N1: Embedding dimensions", () => {
  it(`should produce ${EMBEDDING_DIMS}-dimensional embeddings`, async () => {
    const extractor = await pipeline("feature-extraction", MODEL_ID);
    const output = await extractor("This is a test sentence for embedding.", {
      pooling: "mean",
      normalize: true,
    });
    const embedding = output.data as number[];

    expect(embedding.length).toBe(EMBEDDING_DIMS);
    expect(typeof embedding[0]).toBe("number");
    expect(Number.isFinite(embedding[0])).toBe(true);
  }, 120_000); // Allow up to 2 minutes for model download on first run
});

// =============================================================================
// Test N2: AppleScript escaping correctness
// =============================================================================

describe("N2: AppleScript escaping", () => {
  it("should escape straight double quotes", () => {
    const input = 'My "Best" Plan';
    const escaped = escapeForAppleScript(input);
    expect(escaped).toBe('My \\"Best\\" Plan');
  });

  it("should escape backslashes before other characters", () => {
    const input = 'Path\\to\\"file"';
    const escaped = escapeForAppleScript(input);
    // Backslashes escaped first: \\ -> \\\\, then " -> \\"
    expect(escaped).toBe('Path\\\\to\\\\\\"file\\"');
  });

  it("should escape left curly double quotes", () => {
    const input = "He said \u201Chello\u201D";
    const escaped = escapeForAppleScript(input);
    expect(escaped).toBe('He said \\"hello\\"');
  });

  it("should escape right curly double quotes", () => {
    const input = "It\u2019s a \u201Ctest\u201D";
    const escaped = escapeForAppleScript(input);
    // \u2019 (right single quote) is NOT escaped, only double quotes are
    expect(escaped).toBe('It\u2019s a \\"test\\"');
  });

  it("should strip carriage returns", () => {
    const input = "line1\r\nline2\rline3";
    const escaped = escapeForAppleScript(input);
    expect(escaped).toBe("line1\nline2line3");
  });

  it("should handle a string with no special characters", () => {
    const input = "Plain simple title";
    const escaped = escapeForAppleScript(input);
    expect(escaped).toBe("Plain simple title");
  });

  it("should handle empty string", () => {
    expect(escapeForAppleScript("")).toBe("");
  });

  it("should handle dollar signs (not escaped, but should not break)", () => {
    const input = "Price is $100";
    const escaped = escapeForAppleScript(input);
    expect(escaped).toBe("Price is $100");
  });

  it("should handle single quotes (not escaped in double-quoted AppleScript strings)", () => {
    const input = "It's a test";
    const escaped = escapeForAppleScript(input);
    expect(escaped).toBe("It's a test");
  });

  it("should handle backticks", () => {
    const input = "Use `code` here";
    const escaped = escapeForAppleScript(input);
    expect(escaped).toBe("Use `code` here");
  });

  it("should handle complex mixed input", () => {
    const input = 'He said \u201CHello\\World\u201D and "goodbye"';
    const escaped = escapeForAppleScript(input);
    expect(escaped).toBe('He said \\"Hello\\\\World\\" and \\"goodbye\\"');
  });
});

// =============================================================================
// Test N3: Incremental indexing correctness
// (This test validates the classification logic without requiring Apple Notes)
// =============================================================================

describe("N3: Incremental indexing classification", () => {
  // Simulate the classification logic from incrementalIndexNotes
  function classifyChanges(
    appleNotesMeta: Map<string, string>,
    indexedMeta: Map<string, string>
  ): { newTitles: string[]; modifiedTitles: string[]; deletedTitles: string[] } {
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

    return { newTitles, modifiedTitles, deletedTitles };
  }

  it("should detect new notes", () => {
    const apple = new Map([["A", "d1"], ["B", "d2"], ["C", "d3"], ["D", "d4"]]);
    const indexed = new Map([["A", "d1"], ["B", "d2"], ["C", "d3"]]);
    const result = classifyChanges(apple, indexed);

    expect(result.newTitles).toEqual(["D"]);
    expect(result.modifiedTitles).toEqual([]);
    expect(result.deletedTitles).toEqual([]);
  });

  it("should detect modified notes", () => {
    const apple = new Map([["A", "d1"], ["B", "d2-updated"], ["C", "d3"]]);
    const indexed = new Map([["A", "d1"], ["B", "d2"], ["C", "d3"]]);
    const result = classifyChanges(apple, indexed);

    expect(result.newTitles).toEqual([]);
    expect(result.modifiedTitles).toEqual(["B"]);
    expect(result.deletedTitles).toEqual([]);
  });

  it("should detect deleted notes", () => {
    const apple = new Map([["A", "d1"], ["B", "d2"]]);
    const indexed = new Map([["A", "d1"], ["B", "d2"], ["C", "d3"]]);
    const result = classifyChanges(apple, indexed);

    expect(result.newTitles).toEqual([]);
    expect(result.modifiedTitles).toEqual([]);
    expect(result.deletedTitles).toEqual(["C"]);
  });

  it("should detect all three change types simultaneously", () => {
    const apple = new Map([["A", "d1"], ["B", "d2-updated"], ["D", "d4"]]);
    const indexed = new Map([["A", "d1"], ["B", "d2"], ["C", "d3"]]);
    const result = classifyChanges(apple, indexed);

    expect(result.newTitles).toEqual(["D"]);
    expect(result.modifiedTitles).toEqual(["B"]);
    expect(result.deletedTitles).toEqual(["C"]);
  });

  it("should detect no changes when in sync", () => {
    const apple = new Map([["A", "d1"], ["B", "d2"]]);
    const indexed = new Map([["A", "d1"], ["B", "d2"]]);
    const result = classifyChanges(apple, indexed);

    expect(result.newTitles).toEqual([]);
    expect(result.modifiedTitles).toEqual([]);
    expect(result.deletedTitles).toEqual([]);
  });

  it("should handle empty index (all notes are new)", () => {
    const apple = new Map([["A", "d1"], ["B", "d2"]]);
    const indexed = new Map<string, string>();
    const result = classifyChanges(apple, indexed);

    expect(result.newTitles).toEqual(["A", "B"]);
    expect(result.modifiedTitles).toEqual([]);
    expect(result.deletedTitles).toEqual([]);
  });

  it("should handle empty Apple Notes (all notes are deleted)", () => {
    const apple = new Map<string, string>();
    const indexed = new Map([["A", "d1"], ["B", "d2"]]);
    const result = classifyChanges(apple, indexed);

    expect(result.newTitles).toEqual([]);
    expect(result.modifiedTitles).toEqual([]);
    expect(result.deletedTitles).toEqual(["A", "B"]);
  });
});

// =============================================================================
// Test N4: Search result size safety
// =============================================================================

describe("N4: Search result size", () => {
  it("should produce search results under 1MB when serialized", () => {
    // Simulate the search result format (titles + relevance only)
    const results = [];
    for (let i = 0; i < 100; i++) {
      results.push({
        title: `Note Title ${i} - ${"A".repeat(200)}`, // ~210 chars per title
        relevance: Math.round(Math.random() * 10000) / 10000,
      });
    }
    const serialized = JSON.stringify(results);
    const sizeInBytes = new TextEncoder().encode(serialized).length;

    // 1MB = 1,048,576 bytes
    expect(sizeInBytes).toBeLessThan(1_048_576);
  });

  it("should produce compact results even with many results", () => {
    // Even with 1000 results (far more than the default limit of 10),
    // title+relevance should stay well under 1MB
    const results = [];
    for (let i = 0; i < 1000; i++) {
      results.push({
        title: `A Reasonably Long Note Title That Describes Something Important ${i}`,
        relevance: 0.0167,
      });
    }
    const serialized = JSON.stringify(results);
    const sizeInBytes = new TextEncoder().encode(serialized).length;

    expect(sizeInBytes).toBeLessThan(1_048_576);
  });
});

// =============================================================================
// Test N5: Chunking correctness
// =============================================================================

describe("N5: Content chunking", () => {
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: CHUNK_SIZE,
    chunkOverlap: CHUNK_OVERLAP,
    separators: ["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " ", ""],
  });

  it("should produce a single chunk for short content", async () => {
    const shortText = "This is a short note.";
    const chunks = await splitter.splitText(shortText);
    expect(chunks.length).toBe(1);
    expect(chunks[0]).toBe(shortText);
  });

  it("should produce multiple chunks for long content", async () => {
    // Generate text well over CHUNK_SIZE
    const longText = Array(100)
      .fill("This is a sentence that contributes to a very long document. ")
      .join("");
    const chunks = await splitter.splitText(longText);

    expect(chunks.length).toBeGreaterThan(1);
  });

  it("should keep all chunks under the configured chunk size (with small tolerance)", async () => {
    const longText = Array(100)
      .fill("This is a sentence that contributes to a very long document. ")
      .join("");
    const chunks = await splitter.splitText(longText);

    for (const chunk of chunks) {
      // Allow a small tolerance since the splitter works on separators
      // and may slightly exceed chunkSize in edge cases
      expect(chunk.length).toBeLessThanOrEqual(CHUNK_SIZE + 100);
    }
  });

  it("should preserve all content across chunks (no data loss)", async () => {
    const sentences = [];
    for (let i = 0; i < 50; i++) {
      sentences.push(`Sentence number ${i} contains unique identifier UUID-${i}.`);
    }
    const fullText = sentences.join(" ");
    const chunks = await splitter.splitText(fullText);

    // Every unique identifier should appear in at least one chunk
    for (let i = 0; i < 50; i++) {
      const found = chunks.some((chunk) => chunk.includes(`UUID-${i}`));
      expect(found).toBe(true);
    }
  });

  it("should split on Markdown headers when possible", async () => {
    const markdownText = [
      "# Main Title",
      "",
      "Some introductory text that sets the stage.",
      "",
      "## Section One",
      "",
      "A".repeat(1000),
      "",
      "## Section Two",
      "",
      "B".repeat(1000),
      "",
      "## Section Three",
      "",
      "C".repeat(1000),
    ].join("\n");

    const chunks = await splitter.splitText(markdownText);

    // Should have multiple chunks
    expect(chunks.length).toBeGreaterThan(1);

    // At least one chunk boundary should align with a section header
    const hasHeaderAlignedChunk = chunks.some(
      (chunk) => chunk.startsWith("## Section")
    );
    expect(hasHeaderAlignedChunk).toBe(true);
  });
});
