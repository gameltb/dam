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

Nodes are driven by backend JSON and rendered using a modular architecture.

- **`BaseNode`**: Manages the core container, padding, and the switching logic between modes. Enforces `border-radius: 8px` and ensures content clipping.
- **Modular Renderers**: Specific logic for different media types is encapsulated in standalone components (Image, Video, Markdown, etc.).
- **`GalleryWrapper`**: A reusable container that provides the "Pyramid" expansion logic and right-click context menus for both image and video galleries.
- **`withNodeHandlers` HOC**: Wraps nodes to provide standard handles and resizers.
  - **Selected Visuals**: Uses a fixed `1px` border and a soft blue outer glow (`box-shadow`) to indicate selection without causing layout shifts.
  - **Dynamic Min-Height**: Automatically calculates the required height based on port counts and widget lists to prevent clipping during resize.

### Port & Connection Logic

The system implements a strict, semantic port system.

- **Port Validators**: Connection rules are decoupled into strategies:
  - `StandardValidator`: Single-input, exact type match.
  - `CollectionValidator`: Multiple-inputs, supports "Auto-boxing" (connecting a single element to a List/Set port).
  - `AnyValidator`: Single-input, accepts any data type.
- **Dynamic Guarding**: Real-time feedback during connection dragging:
  - Incompatible ports (wrong type, same side, or full capacity) are dimmed (opacity `0.15`) and grayscale-filtered.
  - `pointer-events` and `isConnectable` are disabled for invalid targets to prevent accidental snapping.

## Communication & State

### Unified Protocol (Protobuf)

The frontend communicates with the backend via a single unified "Envelope" protocol (`FlowMessage`):

- **Incremental Mutations**: Updates are sent as atomic operations (`addNode`, `updateNode`, `removeNode`, `addEdge`).
- **State Hydration**: Uses `nodeUtils.ts` to re-attach complex client-side behavior to raw Protobuf data objects.

### State Management & Synchronization

- **`flowStore` (Zustand + Yjs)**: The central store that manages both the UI state (React Flow nodes/edges) and the collaborative state (`Y.Doc`).
  - It handles the synchronization between the local Zustand store and the shared Yjs document.
  - It directly applies graph mutations to the Yjs document to ensure consistency.
- **Event Bus**: Centralized event bus in `flowStore` for signals like `open-preview` or `open-editor`. **All event names must be defined in a centralized Enum.** Uses a `timestamp` approach to prevent cascading renders in React 19.

### Task System & Optimistic UI

Tracks the lifecycle of long-running operations (`pending` -> `processing` -> `completed`). Uses streaming updates to drive a live progress bar in `ProcessingNode`.

### Layout Engines

- **Global Auto Layout**: Uses `dagre` with a Left-to-Right (`LR`) orientation.
- **Copy/Paste/Duplicate**: Supports subgraph cloning with ID remapping and relative position offsetting.

## Persistence & Quality

- **Zundo**: Provides high-performance undo/redo.
- **Code Quality**: Strict ESLint rules, Prettier formatting, and full TypeScript integration with Protobuf-generated types (zero `any` goal).
