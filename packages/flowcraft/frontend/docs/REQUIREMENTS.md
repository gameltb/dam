# Product Requirements Log

This document tracks the evolving requirements of the Flowcraft application.

## Core Interface

- **Modern Notification Drawer**: A sliding panel from the right for history, with a badge-enabled pill button.
- **Floating Status Panel**: Minimalist pill in the bottom-left showing connection status and URL.
- **Theming**: Full support for Dark and Light modes throughout all components.

## Node System & Media

- **Dynamic Presentation**: Support for "Media Mode" (images, videos, markdown) and "Widgets Mode" (interactive forms).
- **Markdown Support**:
  - Render basic MD syntax (headers, lists).
  - **Inline Editing**: Double-click a Markdown node to enter a text-editor mode; blur to save.
- **Smart Media Preview**:
  - Double-click media nodes to enter web-wide full-screen preview.
  - **Unified Navigation**: Keyboard (Left/Right) and UI buttons work for both images and videos.
  - **Performance**: Preload adjacent gallery items; show loading spinners only for non-cached resources.
  - **Visuals**: Full-screen mode uses sharp corners; close button features hover animations and 90-degree rotation.
- **Asset Editing**: Right-click context menu "Open Editor" opens a dedicated modal placeholder for future asset manipulation tools.

## Layout & Organization

- **Hierarchical Auto Layout**:
  - Treat **Groups as atomic units**: Global layout must not rearrange nodes inside a group relative to the group container.
- **Incremental Placement**: New server-generated nodes must find nearby empty spots using collision detection.
- **Grouping**: Box-selection followed by grouping into a container node.
- **Internal Group Layout**: The ability to auto-arrange only the items inside a specific group.

## Professional Interaction

- **Content Clipping**: Ensure media content does not cover the node's rounded corners (`8px` radius).
- **Input/Output Matching**: Enforce connection types (e.g., only `image` to `image`) unless using `any`.
- **Implicit Connections**: Gray dashed lines for system-generated links (hidden handles).
- **Box Selection**: Left-drag on the canvas to select multiple nodes.
- **Canvas Navigation**: Middle-click drag to pan the canvas.
- **History Management**:
  - `Ctrl+Z` / `Ctrl+Y` support.
  - Versioned state to avoid recording intermediate drag frames.
- **Viewport Persistence**: Backend must remember the user's camera position and zoom.

## Engineering Standards

- **Modular Rendering**: Separate renderer files for each media type to ensure scalability.
- **Decoupled Architecture**: Use a Zustand-based event bus for cross-component communication (e.g., Node -> Preview Modal).
- **Type Safety**: Strictly typed node data and event payloads using TypeScript.
- **Environment**: Prettier and ESLint integrated for consistent code style.
