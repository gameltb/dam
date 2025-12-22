# Backend Design for Flowcraft

This document outlines the design for the Flowcraft backend, focusing on real-time synchronization of the graph editor using WebSockets.

## 1. Technology Stack

-   **Framework**: FastAPI
-   **Server**: Uvicorn
-   **WebSocket Library**: FastAPI's built-in WebSocket support

This stack was chosen for its modern asynchronous capabilities, high performance, and excellent developer experience.

## 2. Architecture: State Management

To manage the WebSocket connections and the authoritative graph state, we will leverage FastAPI's application state and dependency injection system. This avoids the use of global variables and makes the application more robust and testable.

### 2.1. Lifespan Management

The `lifespan` event handler of the FastAPI application will be used to initialize and manage the global state.

-   **On Startup**:
    1.  A `ConnectionManager` instance is created to manage all active WebSocket connections.
    2.  An initial graph state object is created. This object will be a dictionary holding the graph data (nodes and edges) and the current version number.
    3.  Both the `ConnectionManager` and the graph state object are stored in the `app.state` dictionary.

```python
# In packages/flowcraft/src/flowcraft/main.py

from .websockets.main import ConnectionManager

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.connection_manager = ConnectionManager()
    app.state.graph = {
        "data": {"nodes": [], "edges": []},
        "version": 0
    }
    yield
    # Cleanup logic here if needed
```

### 2.2. Dependency Injection

Dependency injection will be used within the WebSocket endpoint to provide access to the managed state in a clean and decoupled way.

-   A `get_connection_manager()` dependency will retrieve the `ConnectionManager` instance from `app.state`.
-   A `get_graph_state()` dependency will retrieve the graph state dictionary from `app.state`.

```python
# In packages/flowcraft/src/flowcraft/websockets/main.py

def get_connection_manager(request: Request) -> ConnectionManager:
    return request.app.state.connection_manager

def get_graph_state(request: Request) -> dict[str, Any]:
    return request.app.state.graph
```

## 3. WebSocket Communication Protocol

All communication between the client and server will be through JSON messages. Each message will have a `type` field to identify its purpose and a `payload` field containing the data.

### 3.1. `sync_graph` (Client -> Server & Server -> Client)

-   **Purpose**: To synchronize the graph state.
-   **Direction**: Bidirectional.
-   **Payload Structure**:
    ```json
    {
      "version": 12,
      "graph": {
        "nodes": [...],
        "edges": [...]
      }
    }
    ```

### 3.2. `execute_action` (Client -> Server)

-   **Purpose**: For the client to request the execution of a dynamic action on a specific node.
-   **Direction**: Client to Server.
-   **Payload Structure**:
    ```json
    {
      "actionId": "generate-children",
      "nodeId": "abc-123"
    }
    ```

### 3.3. `apply_changes` (Server -> Client)

-   **Purpose**: For the server to push a set of changes to the client after an action has been executed.
-   **Direction**: Server to Client.
-   **Payload Structure**:
    ```json
    {
      "add": [
        // new node objects
      ],
      "update": [
        // partial node objects with id
      ],
      "remove": [
        // array of node ids to remove
      ]
    }
    ```

## 4. Version Control Mechanism

To prevent race conditions and ensure data consistency, a simple versioning system is implemented.

1.  The server maintains an authoritative `version` number for the graph, starting at `0`.
2.  When a client connects, the server sends the current graph and its `version`.
3.  The client stores this `version` number.
4.  When the client sends a `sync_graph` message, it **must** include the `version` number it currently has.
5.  **On the server**:
    -   If the client's `version` matches the server's `version`, the server accepts the new graph state, increments its own `version` number, and broadcasts the updated graph and new version to all connected clients.
    -   If the client's `version` **does not match**, it signifies a conflict (another client has updated the state). The server rejects the client's change and sends back the current authoritative graph and version, along with an `error: "version_mismatch"` field. The client is then expected to overwrite its local state with this authoritative version.

## 5. Dynamic Action Flow

1.  **Client**: A user selects a node and clicks an action button in the context menu.
2.  **Client**: An `execute_action` message is sent to the server.
3.  **Server**: The WebSocket endpoint receives the message. It identifies the action and the target node.
4.  **Server**: The corresponding business logic is executed. This logic calculates the necessary changes to the graph (e.g., creating new nodes, updating properties).
5.  **Server**: A changeset is constructed.
6.  **Server**: An `apply_changes` message containing the changeset is broadcast to all clients.
7.  **Client**: Each client receives the `apply_changes` message and updates its local Zustand store accordingly. For new nodes, the client is responsible for computing their initial positions (e.g., via an auto-layout algorithm).
8.  **Client**: After the changes are applied and the view is updated, the existing `useEffect` hook in the client will automatically trigger a `sync_graph` message, sending the final, complete graph state (including the new node positions) back to the server for persistence.
