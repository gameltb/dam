# Product Requirements Log

This document tracks the evolving requirements of the Flowcraft application.

## Core Interface

- **Modern Notification Drawer**: A sliding panel from the right for history, with a badge-enabled pill button.
- **Floating Status Panel**: Minimalist pill in the bottom-left showing connection status and URL.
- **Theming**: Full support for Dark and Light modes throughout all components, including **Right-Click Context Menus**.

## Node System & Media

- **Dynamic Presentation**: Support for "Media Mode" (images, videos, markdown) and "Widgets Mode" (interactive forms).
- **Markdown Support**: Render basic MD syntax with inline double-click editing.
- **Advanced Interaction**:
  - **Streaming Outputs**: Real-time "typing" effects for AI responses.
  - **Long-Running Tasks**: Background jobs with progress bars and cancellation support.
- **Visual Stability**:
  - **Selected State**: Selecting a node must NOT change its dimensions or cause internal layout shifts (use fixed border + glow).
  - **Auto-Resize Guards**: Nodes must have a calculated minimum height based on their ports and widgets.

## Port & Connection System

- **Differentiated Logic**:
  - **Standard Ports**: Exactly one input allowed.
  - **Collection Ports (List/Set)**: Support multiple inputs of matching types.
  - **Any Ports**: Accept a single input of any type.
- **Dynamic Guarding (UX)**:
  - **Active Drag Feedback**: While dragging a wire, incompatible target ports must dim and apply a grayscale filter.
  - **Snapping Guard**: Incompatible ports must disable snapping and pointer interactions to prevent illegal connections.
  - **Tooltip Metadata**: Ports must show type info and connection limits on hover.

## Layout & Organization

- **Hierarchical Auto Layout**:
  - Global `dagre` layout using Left-to-Right orientation.
  - Treat **Groups as atomic units** during layout.
- **Clipboard Operations**:
  - **Copy/Paste/Duplicate**: Support for selecting multiple nodes and edges.
  - **ID Remapping**: Pasted subgraphs must have new unique IDs while maintaining internal connectivity.
  - **Position Awareness**: Pasting should occur at the mouse cursor position or with a consistent offset.

## Professional Interaction

- **Implicit Connections**: Gray dashed lines for system-generated or widget-bound links.
- **History Management**: `Ctrl+Z` / `Ctrl+Y` with frame-skipping for drag performance.
- **Viewport Persistence**: Remember camera position and zoom across sessions.

## Engineering Standards

- **Unified Protocol**: All communications must use the Protobuf-defined `FlowMessage` envelope.
- **Type Safety**: Strictly typed node data; elimination of `any` in core components.
- **React 19 Readiness**: Ensure all state updates within effects are deferred or handled asynchronously to avoid rendering warnings.
