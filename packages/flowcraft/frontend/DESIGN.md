# Frontend Design Documentation

This document outlines the design of the frontend components for the Flowcraft application, with a focus on the node-based interface.

## Node Component Architecture

The core of the UI is a node-based editor built with `@xyflow/react`. To support a variety of node types and functionalities, we've implemented a flexible and extensible node component architecture.

### `BaseNode` Component

The foundation of our node system is the `BaseNode` component (`src/components/base/BaseNode.tsx`). It serves as a generic container for all nodes and is responsible for:

- Managing the rendering mode of the node.
- Rendering the node's content based on the current mode.
- Providing a `toggleMode` function to its children to allow them to trigger a mode change.

### `withNodeHandlers` High-Order Component

To reduce boilerplate code in our node components, we've created a High-Order Component (HOC) called `withNodeHandlers` (`src/components/hocs/withNodeHandlers.tsx`). This HOC is responsible for:

- Wrapping the `BaseNode` component.
- Providing the `renderMedia` and `renderWidgets` functions to the `BaseNode`.
- Handling the type assertions for the `renderMedia` and `renderWidgets` functions.
- Rendering the input and output handles for the node.

### Rendering Modes

Nodes can have two primary rendering modes:

1.  **Media Mode**: Optimized for viewing media content such as images, videos, and audio. In this mode, the media is displayed prominently within the node.
2.  **Widgets Mode**: A vertical stack of interactive UI elements (widgets) like text fields, select dropdowns, and buttons. This mode is used for editing and configuring the node's data.

A node can support either one or both of these modes. The `WidgetWrapper` component provides a button and a context menu to switch between modes if the node is switchable.

### Widgets

Widgets are reusable React components that provide specific UI functionalities. They are located in the `src/components/widgets` directory. Examples include:

- `TextField`: A simple text input field.
- `SelectField`: A dropdown menu for selecting from a list of options.

Each widget is wrapped in a `WidgetWrapper` component, which provides:

- A visual selection indicator.
- A context menu for mode switching.
- A button for toggling the rendering mode.

### Interface Type System

To ensure that nodes are connected in a meaningful way, we have implemented a type system for the node handles (the connection points).

- The `onConnect` function in `App.tsx` checks if the `outputType` of the source node and the `inputType` of the target node match before creating a connection. It allows connections if the types match, or if either type is `"any"`. This prevents invalid connections between nodes.
- The `inputType` and `outputType` are defined in the node's data type (e.g., `ImageNodeData`).

## Creating a New Node

To create a new custom node, follow these steps:

1.  **Define the Node Data**: Create a new type definition for your node's data. It should include an `outputType` and `inputType` property.

2.  **Create the Rendering Functions**:
    - Create a `renderMedia` function to define how your node should be displayed in media mode. If your node does not have a media view, you can pass an empty function.
    - Create a `renderWidgets` function to define how your node should be displayed in widgets mode.

3.  **Create the Node Component**:
    - Create a new `.tsx` file in the `src/components` directory.
    - Use the `withNodeHandlers` HOC to create your node component, passing in the `renderMedia` and `renderWidgets` functions.

4.  **Register the Node Type**: In `App.tsx`, add your new node component to the `memoizedNodeTypes` object. This makes the new node type available to the React Flow instance.

By following this architecture, we can easily create new node types with custom UIs and behaviors while maintaining a consistent and predictable user experience.
