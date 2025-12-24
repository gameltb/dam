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
- **Modular Renderers**: Specific logic for different media types is encapsulated in standalone components:
  - `ImageRenderer`: Standard image display.
  - `VideoRenderer`: Supports auto-play, muting, and full controls.
  - `MarkdownRenderer`: Supports rich-text preview and double-click-to-edit inline mode.
- **`GalleryWrapper`**: A reusable container that provides the "Pyramid" expansion logic and right-click context menus for both image and video galleries.
- **`withNodeHandlers` HOC**: Wraps nodes to provide standard handles, resizers, and event-bus integration.

### Rendering Modes

1.  **Media Mode**:
    - Optimized for images/videos/markdown.
    - **Interactive Preview**: Double-clicking a media node enters full-screen preview.
    - **Gallery**: Supports stable triangular expansion. Right-click extraction allows turning any gallery item into a standalone node.
2.  **Widgets Mode**:
    - Vertical stack of interactive elements (`TextField`, `SelectField`, `CheckboxField`, `SliderField`, `Button`).
    - **Two-way Binding**: Changes are synced back to the server state immediately.

## Communication & State

### Event Bus (Zustand)

Instead of deep prop drilling or global DOM events, the system uses a centralized event bus in `flowStore`:

- **`dispatchNodeEvent`**: Used by nodes to signal intents (e.g., `open-preview`, `open-editor`).
- **`lastNodeEvent`**: A state tracked with a `timestamp` to allow components to respond precisely once to user actions without cascading renders.

### Layout Engines

- **Global Auto Layout**: Uses `dagre`. Refined to treat **GroupNodes** as single atomic units, preserving the internal relative layout of children.
- **Incremental Layout**: Positions new nodes using collision detection to find nearby empty space without disturbing manual arrangements.
- **Internal Group Layout**: Manually triggerable layout for nodes strictly inside a specific container.

## Persistence & Quality

- **Zundo**: Provides undo/redo with version-based frame skipping for performance.
- **Node Hydration**: Centralized utility (`nodeUtils.ts`) to re-attach client-side handlers to server-provided JSON nodes.
- **Code Quality**: Strict ESLint rules, Prettier formatting, and comprehensive TypeScript typing (avoiding `any` in favor of `unknown` and generics).
