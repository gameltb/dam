# Frontend Design Documentation

This document outlines the design of the frontend components for the Flowcraft application.

## Application Shell & UI/UX

The application uses a modern, dark-mode-first aesthetic with a "floating" UI approach and highly responsive modals.

### Status Panel

- **Location**: Bottom-left. Displays connectivity and current backend URL.
- **Function**: Clickable to edit URL or manually trigger a sync if out of date.

### Notification System

- **Components**: A pill-shaped toggle button (top-right) and a sliding **Drawer**.
- **Features**: Persistent history, themed styling, and "newest-first" sorting.

### Global Modals (Event-Driven)

- **Media Preview**: A high-performance, full-screen overlay for viewing images and videos.
- **Editor Placeholder**: A dedicated space for future integrated asset editors (crop, filters, etc.).

## Node Component Architecture

The core is a dynamic node-based editor built with `@xyflow/react` using a decoupled, event-driven approach.

### `DynamicNode` System

Nodes are driven by backend Protobuf schemas and rendered using a modular architecture.

- **`NodeShell` (Foundation)**: Manages the core container, padding, and provides node identity via **`NodeProvider` (Context API)**. This eliminates `nodeId` prop-drilling for all internal components.
- **Viewport-Driven Hydration**: Implements lazy-rendering for complex node contents using `IntersectionObserver`. Node frames are rendered immediately to maintain graph topology, while expensive children are only hydrated when entering the viewport.
- **Modular Renderers**: Specific logic for different media types is encapsulated in standalone components (Image, Video, Markdown, etc.).
- **`useNodeProperty` Hook**: Provides declarative, two-way binding for specific node data fields, abstracting away the sync pipeline.

### Port & Connection Logic

The system implements a strict, semantic port system.

- **Port Validators**: Connection rules are decoupled into strategies:
  - `StandardValidator`: Single-input, exact type match.
  - `CollectionValidator`: Multiple-inputs, supports "Auto-boxing".
  - `AnyValidator`: Single-input, accepts any data type.
- **Dynamic Guarding**: Real-time feedback during connection dragging with visual dimming and grayscale filters for incompatible targets.

## Communication & State

### Unified Mutation Pipeline (Immer Patches)

The system utilizes a modern, patch-based architecture for all state changes:

- **ID-Based Master State**: The store maintains nodes and edges in `nodesById` and `edgesById` maps (`Record<string, T>`). This ensures O(1) access and stable, ID-driven patch paths.
- **Automated Patch Generation**: All changes are applied via `applyMutations(recipe)`. Using Immer's `produceWithPatches`, the system automatically generates fine-grained `patches` (for sync) and `inversePatches` (for undo).
- **Middleware Pipeline**: Mutations flow through a sequence of middlewares:
  - **`HistoryMiddleware`**: Captures inverse patches for the undo stack.
  - **`TaskMiddleware`**: Logs mutations against active background tasks.
  - **`SyncMiddleware`**: Translates patches into SpacetimeDB Reducer calls.

### Mirror Repository Synchronization

Synchronization is handled as a reconciliation process between the local store and remote tables:

- **`GraphMapper`**: Centralized logic for converting between SpacetimeDB Rows and Frontend Domain Objects.
- **Atomic Reconciliation**: `useGraphSync` performs O(N) diffing between remote tables and `nodesById`, ensuring high performance even with thousands of nodes.
- **Conflict Isolation**: Remote updates are ignored for nodes currently being interacted with (`isInteracting`) to prevent UI flickering.

### Unified Protocol (Protobuf v2)

- **Strict Typing**: Uses native TypeScript Discriminated Unions for `AppNode` sub-types.
- **Standard Serialization**: Uses Buf's `toJsonString` and `fromJsonString` for all message persistence (e.g., `parts_json`), ensuring strict compliance with Protobuf specs.

## Persistence & Quality

- **Zustand**: Global state management with custom history middleware.
- **Code Quality**: Strict ESLint rules, English-only comments, and full TypeScript integration with Zero-`any` goal.
