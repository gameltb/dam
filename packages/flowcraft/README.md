# FlowCraft

FlowCraft is a real-time, collaborative, node-based UI.

## Features

- Real-time synchronization of graph data using WebSockets.
- Custom text and image nodes with editable content.
- "Focus view" feature for isolating and editing a subgraph.
- Integration with the `dam` package to display entity and component data.

## Setup

### Backend

To run the backend server, navigate to the root of the repository and run the following command:

```bash
uv run poe dev --package flowcraft
```

The server will be available at `http://127.0.0.1:8000`.

### Frontend

To run the frontend development server, navigate to the `packages/flowcraft/frontend` directory and run the following command:

```bash
npm run dev
```

The application will be available at `http://localhost:5173`.
