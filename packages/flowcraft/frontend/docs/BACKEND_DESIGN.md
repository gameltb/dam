# Backend Design Documentation

This document outlines the design of the mock backend for Flowcraft.

## Overview

The backend is simulated using **MSW (Mock Service Worker)**, allowing the frontend to interact with a realistic REST API that maintains an in-memory state.

## State Management

The "Server" state is held in `src/mocks/handlers.ts`:

- `serverGraph`: Stores the list of nodes, edges, and the current viewport.
- `serverVersion`: A monotonic counter used for conflict detection and version matching.

## Protocol & Endpoints

The system uses a hybrid communication strategy:

1.  **REST/HTTP**: For resource fetching (templates).
2.  **Protobuf/RPC** (Logical): For strict graph synchronization and commands.
3.  **Streaming**: For AI generation and long-running tasks.

### 1. Protobuf Schema (`flowcraft.proto`)

All data structures are strictly defined in `schema/flowcraft.proto`. This acts as the contract between frontend and backend.
Key Definitions:

- `Graph`: The full snapshot state.
- `NodeData`: The schema-driven definition of a node's capabilities (modes, media, widgets).
- `TaskRequest` / `TaskUpdate`: The standard envelope for job execution.

### 2. Synchronization (`/api/graph`)

- **GET**: Polling for full graph state (proto-compliant JSON).
- **POST**: Pushing changes.

### 3. Widget Interaction Service

- **Real-time Updates**: `/api/widget/update` for lightweight value syncing (e.g., slider dragging).
- **Dynamic Options**: `/api/widget/options` allows Select widgets to fetch data from the server (e.g., listing available AI models).

### 4. Task Execution System (`/api/task/*`)

For long-running operations (like "Analyze Data" or "Generate Image"), we use a Job System:

1.  **Execute** (`POST /api/task/execute`):
    - Client sends `TaskRequest` (Task Type + Params).
    - Server responds with a **Chunked Stream** (application/x-ndjson or text/event-stream).
    - Client receives `TaskUpdate` events (status: PENDING -> PROCESSING -> COMPLETED).
2.  **Cancel** (`POST /api/task/cancel`):
    - Client can abort a running task by ID.

## Node Definitions (JSON/Proto)

Nodes are defined dynamically. A typical node definition includes:

- `modes`: List of supported rendering modes.
- `media`: Image/Video info + `aspectRatio` + `gallery` (optional array of strings).
- `widgets`: Array of widget definitions (type, label, value, options).
