# Backend Design Documentation

This document outlines the design of the mock backend for Flowcraft.

## Overview

The backend is simulated using **MSW (Mock Service Worker)**, allowing the frontend to interact with a realistic REST API that maintains an in-memory state.

## State Management

The "Server" state is held in `src/mocks/handlers.ts`:

- `serverGraph`: Stores the list of nodes, edges, and the current viewport.
- `serverVersion`: A monotonic counter used for conflict detection and version matching.

## Protocol & Endpoints

### 1. Node Creation System

The backend defines what can be created via the `GET /api/node-templates` endpoint.

- **Templates**: Define the default data (widgets, media, types) and the **Menu Path** (hierarchy).
- This allows the backend to reorganize the frontend's creation menu without any frontend code changes.

### 2. Synchronization (`/api/graph`)

- **GET**: Used for polling. Returns the full graph state and version.
- **POST**: Used by the client to push changes. Increments the version upon success.

### 3. Server Actions (`/api/action`)

- Simulates server-side logic like child generation.
- Instead of returning a full graph, it returns a **Diff** (nodes and edges to add/update), which the frontend merges using `Incremental Layout`.

## Node Definitions (JSON)

Nodes are defined dynamically. A typical node definition includes:

- `modes`: List of supported rendering modes.
- `media`: Image/Video info + `aspectRatio` + `gallery` (optional array of strings).
- `widgets`: Array of widget definitions (type, label, value, options).
