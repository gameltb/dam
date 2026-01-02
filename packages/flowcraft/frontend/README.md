# Flowcraft

Flowcraft is a high-performance, backend-driven node-based editor built with React 19, `@xyflow/react` (React Flow 12), and TypeScript. It features a unified incremental update protocol using Protobuf, real-time collaboration support via Yjs, and a robust task system for long-running operations.

## Architecture & Core Technologies

- **Frontend:** React 19, Vite, TypeScript
- **Graph Engine:** `@xyflow/react` (React Flow 12)
- **State Management:** Zustand (global state), Zundo (undo/redo), Yjs (CRDT/Collaboration)
- **Protocol:** Protocol Buffers (Protobuf) for data serialization and contracts
- **Layout:** `dagre` for automatic directed graph positioning
- **Mocking:** Mock Service Worker (MSW) for simulating a stateful backend

## Key Features

### 1. Dynamic Node Architecture

Nodes are defined by backend JSON/Protobuf schemas.

- **Media Mode:** Renders images, video, and markdown.
- **Widgets Mode:** Interactive fields like sliders, selects, and inputs.
- **Implicit Ports:** Ports can be tied to widgets, enabling dynamic connections.

### 2. Unified Protocol

All graph operations (snapshots, mutations, task updates) use a single `FlowMessage` envelope. Updates are incremental and atomic (e.g., `addNode`, `updateNode`).

### 3. Task System

Supports long-running operations (e.g., AI generation) with real-time lifecycle tracking (`pending` -> `processing` -> `completed`).

### 4. Development Standards

- **Strict Typing:** No `any`. Uses generated Protobuf types.
- **Testing:** Comprehensive Vitest setup with regression coverage for architectural decisions.
- **Linting:** ESLint with strict type-checking and Prettier integration.

## Getting Started

### Prerequisites

- Node.js (v20+ recommended)
- npm

### Installation

```bash
npm install
```

### Development Server

Starts the Vite dev server and the MSW worker.

```bash
npm run dev
```

### Building

```bash
npm run build
```

### Verification

Run the full quality pipeline (Protobuf generation, type check, linting, testing, build):

```bash
npm run verify
```

## Project Structure

- `src/components`: UI components organized by domain (nodes, edges, widgets).
- `src/store`: State management (Zustand stores).
- `src/generated`: Protobuf generated TypeScript files.
- `src/mocks`: MSW handlers and mock data.
- `src/utils`: Helper functions for nodes, ports, and Protobuf adaptation.
- `docs/`: Detailed design and architecture documentation.

## Documentation

For more details on the architecture, see [docs/DESIGN.md](docs/DESIGN.md).
