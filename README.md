# MCP Apple Notes

![MCP Apple Notes](images/logo.png)

A [Model Context Protocol (MCP)](https://www.anthropic.com/news/model-context-protocol) server that enables semantic search and RAG (Retrieval Augmented Generation) over your Apple Notes. This allows AI assistants like Claude to search and reference your Apple Notes during conversations.

![MCP Apple Notes](images/demo.png)

> **Fork note:** This is a fork of [RafalWilinski/mcp-apple-notes](https://github.com/RafalWilinski/mcp-apple-notes) via [Tom-Semple/mcp-apple-notes-fixed](https://github.com/Tom-Semple/mcp-apple-notes-fixed), with significant improvements to reliability, search quality, and indexing architecture.

## Features

- 🔍 Semantic search over Apple Notes using [`bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) on-device embeddings (384-dim, 512-token context)
- 📝 Full-text search capabilities with hybrid vector + FTS ranking via reciprocal rank fusion (RRF)
- 📊 Vector storage using [LanceDB](https://lancedb.github.io/lancedb/)
- 🤖 MCP-compatible server for AI assistant integration
- 🍎 Native Apple Notes integration via AppleScript (`osascript`)
- 🏃‍♂️ Fully local execution, all data stays on-device, no API keys needed
- 🔄 Incremental indexing: only new, modified, and deleted notes are re-processed
- ✂️ Content chunking: long notes are split into overlapping chunks so no content is lost during embedding
- 🚀 Auto-indexes on MCP server startup with graceful failure handling

## What changed from the original

| Feature | Original | This fork |
|---|---|---|
| Embedding model | `all-MiniLM-L6-v2` (22M params, 256-token context) | `bge-small-en-v1.5` (33M params, 512-token context) |
| Apple Notes access | JXA via `run-jxa` (times out on large libraries) | AppleScript via `osascript`; metadata fetched in a single bulk call, content fetched per-note only when needed |
| Indexing | Full rebuild every time, background batch processing | Incremental: only re-embeds new/modified notes, removes deleted |
| Content handling | Whole note as single embedding (truncated at context window) | Markdown-aware chunking with overlap (no content lost) |
| Search results | Returns full content (exceeds 1MB MCP limit on large libraries) | Returns titles + relevance scores (use `get-note` for full content) |
| Index freshness | Manual trigger required | Auto-indexes on server startup |
| Input sanitization | Basic quote escaping | Dedicated `escapeForAppleScript()` for injection prevention |
| Logging | Ad-hoc `console.error` | Structured logging with timestamps and levels to stderr |
| Error handling | Server crashes if Apple Notes is unavailable | Graceful startup: logs warning, starts server, user can retry |

## Prerequisites

- macOS (Apple Notes is macOS-only)
- [Bun](https://bun.sh/docs/installation)
- [Claude Desktop](https://claude.ai/download) or another MCP-compatible client

## Installation

1. Clone the repository:

```bash
git clone https://github.com/mcmaster-aaron/mcp-apple-notes-fixed.git
cd mcp-apple-notes-fixed
```

2. Install dependencies:

```bash
bun install
```

3. Build the initial search index (required before first use):

```bash
bun run index-cli.ts
```

This fetches all your Apple Notes via AppleScript, converts HTML to Markdown, chunks long notes, generates embeddings with `bge-small-en-v1.5`, and stores everything in LanceDB at `~/.mcp-apple-notes/data`. The first run also downloads the embedding model (~80MB, cached for subsequent runs).

For large note collections (1000+), this may take several minutes. Progress is logged to the console.

## Usage

### Configure Claude Desktop

1. Open Claude Desktop and go to **Settings > Developer > Edit Config**

![Claude Desktop Settings](images/desktop_settings.png)

2. Open `claude_desktop_config.json` and add the following entry:

```json
{
  "mcpServers": {
    "local-machine": {
      "command": "/Users/<YOUR_USER_NAME>/.bun/bin/bun",
      "args": ["/Users/<YOUR_USER_NAME>/mcp-apple-notes-fixed/index.ts"]
    }
  }
}
```

> **Important:** Replace `<YOUR_USER_NAME>` with your actual macOS username. Update the path if you cloned the repo to a different location.

3. Restart Claude Desktop. You should see the Apple Notes tools available:

![Claude MCP Connection Status](images/verify_installation.png)

### How it works

On startup, the MCP server automatically runs an incremental index update. It compares the `modification_date` of every note in Apple Notes against what's stored in the LanceDB index, then only re-processes notes that are new, modified, or deleted. If Apple Notes is unavailable (e.g., during iCloud sync), the server starts anyway and uses the existing index.

### Available tools

| Tool | Description |
|---|---|
| `search-notes` | Semantic + full-text hybrid search. Returns ranked note titles with relevance scores. |
| `get-note` | Fetch the full content of a specific note by title. |
| `list-notes` | List the total number of indexed note chunks. |
| `index-notes` | Manually trigger an incremental index update. |
| `create-note` | Create a new Apple Note with the given title and HTML content. |

### Typical workflow

1. Ask Claude to search your notes: *"Search my notes for cocktail recipes with green chartreuse"*
2. Claude calls `search-notes` and gets back a ranked list of note titles.
3. Claude calls `get-note` on the most relevant titles to read the full content.
4. Claude synthesizes an answer from the retrieved notes.

## CLI indexer

The `index-cli.ts` script can be used independently of the MCP server for index management.

### Full rebuild

Drops and recreates the entire index from scratch. Use for initial setup or recovery:

```bash
bun run index-cli.ts
```

### Incremental update

Only processes new, modified, and deleted notes. Much faster for day-to-day use:

```bash
bun run index-cli.ts --incremental
```

### npm scripts

```bash
bun run index             # Full rebuild
bun run index:incremental # Incremental update
bun run test              # Run test suite
bun run purge-db          # Delete the entire index (rm -rf ~/.mcp-apple-notes)
```

## Optional: scheduled index rebuilds

If you want to keep the index fresh without relying solely on MCP server restarts, you can schedule the CLI indexer to run periodically via macOS `launchd`.

A sample plist is included at `com.mcp-apple-notes.index.plist`. To install:

```bash
cp com.mcp-apple-notes.index.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.mcp-apple-notes.index.plist
```

Verify it's loaded:

```bash
launchctl list | grep mcp-apple-notes
```

Expected output:

```
-	0	com.mcp-apple-notes.index
```

To unload:

```bash
launchctl unload ~/Library/LaunchAgents/com.mcp-apple-notes.index.plist
```

> **Tip:** Update the plist to pass `--incremental` for faster daily runs. The default runs a full rebuild.

## Testing

```bash
bun test
```

The test suite includes:

- **Embedding dimension validation**: Confirms the model produces 384-dimensional vectors.
- **AppleScript escaping**: Tests that special characters (double quotes, backslashes, smart quotes, etc.) are safely escaped to prevent injection.
- **Incremental indexing logic**: Validates correct classification of new, modified, and deleted notes.
- **Search result size**: Asserts that search responses stay under the 1MB MCP response limit.
- **Content chunking**: Verifies chunk sizes, overlap, content preservation, and Markdown-aware splitting.
- **Bulk metadata parsing**: Validates the `title|||modDate~~~` format parser used by the single-call AppleScript metadata fetch.
- **Note meta parsing**: Validates the `title|||creation_date|||mod_date` format parser used when fetching individual note details.

> **Note:** The embedding dimension test downloads the model on first run and may take up to 2 minutes.

## Troubleshooting

### Viewing logs

The MCP server logs to stderr with structured timestamps:

```bash
tail -n 50 -f ~/Library/Logs/Claude/mcp-server-local-machine.log
# or
tail -n 50 -f ~/Library/Logs/Claude/mcp.log
```

CLI indexer logs to stdout:

```bash
bun run index-cli.ts 2>&1 | tee ~/.mcp-apple-notes/index-cli.log
```

### Apple Notes AppleEvent timeouts

If you see `AppleEvent timed out` errors, Apple Notes is likely busy with iCloud sync or is unresponsive. Try:

1. Open Apple Notes and wait for sync to complete.
2. If still stuck, restart Apple Notes:
   ```bash
   osascript -e 'tell application "Notes" to quit'
   sleep 5
   open -a Notes
   ```
3. Wait 30 seconds, then retry.

### Index corruption or stale data

If search results seem wrong or the index is in a bad state, do a full rebuild:

```bash
bun run purge-db
bun run index-cli.ts
```

Then restart the MCP server.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────┐
│  Claude Desktop  │◄───►│   MCP Server     │◄───►│ Apple Notes │
│  (MCP Client)    │     │   (index.ts)     │     │ (osascript) │
└─────────────────┘     └────────┬─────────┘     └─────────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │     LanceDB      │
                        │ (~/.mcp-apple-   │
                        │   notes/data)    │
                        │                  │
                        │  Vector index    │
                        │  + FTS index     │
                        │  (bge-small-     │
                        │   en-v1.5)       │
                        └──────────────────┘
```

### Key design decisions

- **On-device only.** No note content ever leaves your machine. The embedding model runs locally via Transformers.js / ONNX Runtime.
- **Incremental by default.** The MCP server indexes on startup, and only processes what changed. For a 1,500-note collection where a handful change between restarts, this takes seconds instead of minutes.
- **Chunked embeddings.** Long notes are split into overlapping chunks (1,500 chars with 200-char overlap) using Markdown-aware splitting. Each chunk is a separate row in LanceDB sharing the parent note's title and modification date. This ensures every part of every note is searchable.
- **Two-stage search.** `search-notes` returns lightweight title + relevance results (never exceeds 1MB). `get-note` fetches full content from Apple Notes directly (not from the index). This decouples search from retrieval and avoids the MCP response size limit.
- **Bulk AppleScript metadata fetch.** All note titles and modification dates are fetched in a single `osascript` call that iterates folders and notes internally, returning a delimited string parsed in JS. This replaces the prior per-note approach (which spawned one process per note plus a 100ms sleep between each). Full note content (HTML body) is still fetched per-note when re-indexing is required.

## License

See upstream: [RafalWilinski/mcp-apple-notes](https://github.com/RafalWilinski/mcp-apple-notes)
