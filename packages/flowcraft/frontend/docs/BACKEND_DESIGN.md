# Backend Design Documentation

This document outlines the design of the mock backend for Flowcraft.

## Overview

The backend is simulated using a hybrid approach:
1.  **MSW (Mock Service Worker)**: For legacy REST endpoints (like `/api/node-templates`).
2.  **ConnectRPC (gRPC-web compatible)**: For the core Flowcraft service. This provides a type-safe, contract-first API using Protobuf.

The frontend communicates with this mock service via an in-memory `routerTransport`, which can be easily swapped for a real network transport (Connect, gRPC-web, or gRPC).

## State Management

The "Server" state is held in `src/mocks/db.ts`:

- `serverGraph`: Stores the list of nodes, edges, and the current viewport.
- `serverVersion`: A monotonic counter used for conflict detection and version matching.

## Protocol & Implementation

### 1. FlowService Implementation

The core logic is implemented in `src/mocks/flowServiceImpl.ts`. This implementation handles:
- **watchGraph**: A server-streaming RPC that sends an initial snapshot and then streams incremental mutations, task updates, and widget signals.
- **applyMutations**: Pushes graph changes from the client to the server.
- **executeAction**: Triggers long-running background tasks.
- **updateNode/updateWidget**: Specialized mutations for frequent updates.

### 2. Event Bus (`mockEventBus.ts`)

Since multiple RPC calls (like `applyMutations` and `watchGraph`) need to coordinate, a central `mockEventBus` is used to broadcast events within the mock backend environment.

### 3. Protobuf Schema (`schema/`)

All data structures are strictly defined in `.proto` files.
Key Definitions:
- `FlowService`: The service definition with all RPC methods.
- `NodeSchema`: The schema-driven definition of a node's capabilities.
- `GraphMutation`: Atomic operations for graph synchronization.


## Node Definitions (JSON/Proto)

Nodes are defined dynamically. A typical node definition includes:

- `modes`: List of supported rendering modes.
- `media`: Image/Video info + `aspectRatio` + `gallery` (optional array of strings).
- `widgets`: Array of widget definitions (type, label, value, options).
