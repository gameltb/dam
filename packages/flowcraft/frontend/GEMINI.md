# Flowcraft Project Context

Flowcraft is a high-performance, backend-driven node-based editor built with React and `@xyflow/react` (React Flow 12). It features a unified incremental update protocol using Protobuf and a robust task system for long-running operations.

## Architecture & Core Technologies

- **Frontend:** React 19, Vite, TypeScript.
- **Graph Engine:** `@xyflow/react` for node and edge management.
- **State Management:** Zustand for global state, Zundo for undo/redo functionality.
- **Layout:** `dagre` for automatic directed graph positioning.
- **Protocol:** Protocol Buffers (Protobuf) for data serialization and communication contracts.
- **Mocking:** Mock Service Worker (MSW) simulates a stateful backend.
- **Messaging:** Unified "Envelope" protocol (`FlowMessage`) via HTTP streaming to simulate WebSocket behavior.

## Key Systems

### 1. Node Architecture

Nodes are dynamic and driven by backend-defined JSON/Protobuf schemas.

- **DynamicNode**: A modular component that can switch between **Media Mode** (images, video, markdown) and **Widgets Mode** (interactive fields like sliders, selects, etc.).
- **ProcessingNode**: A temporary placeholder for ongoing tasks, providing live progress feedback and cancellation.
- **Port System**: Supports explicit and implicit ports. Implicit ports are tied to widgets and can toggle a widget's enabled state when connected.

### 2. Protocol & Communication

- **FlowMessage Envelope**: All communications (snapshots, mutations, task updates, streaming chunks) are wrapped in a unified Protobuf message.
- **Incremental Mutations**: The graph state is updated via atomic mutations (`addNode`, `updateNode`, `removeNode`, `addEdge`, etc.), ensuring efficient synchronization.
- **Hydration**: Client-side handlers are re-attached to server-provided node data using a centralized hydration utility.

### 3. Task & Job System

- Long-running operations (e.g., AI generation) use a task store to track lifecycles (`pending`, `processing`, `completed`, `failed`).
- Real-time updates are streamed from the mock backend to the frontend.

## Building and Running

- **Development:** `npm run dev` (A persistent development server is usually running at `http://localhost:5173/`. You can use Chrome DevTools to inspect the page).
- **Build:** `npm run build`
- **Linting:** `npm run lint`
- **Formatting:** `npm run format`
- **Protobuf Generation:** `npm run proto:generate` (Requires `schema/*.proto` files)

## Development Conventions

- **Typing:** Strict TypeScript typing is enforced. Avoid `any` in favor of `unknown`, generics, or generated Protobuf types.
- **Event Bus:** Use the centralized event bus in `flowStore` (`dispatchNodeEvent`) for inter-component signaling (e.g., opening previews or editors).
- **Theming:** The application is dark-mode first. Use CSS variables defined in `index.css` (`--node-bg`, `--primary-color`, etc.) for all styling.
- **Components:** Modularize node renderers and widgets. Prefer Functional Components and Hooks.
- **Mocking:** When adding new API features, update the MSW handlers in `src/mocks/` to maintain a consistent local development experience.

## Quality & Development Workflow

To maintain high code quality and prevent regressions, follow this iterative process for every feature or bug fix:

1.  **Analyze & Reproduce:** Understand the requirement or bug.
2.  **Write Tests First (or during):** Create a test file in `__tests__/` that explicitly describes the **Problem** and the **Requirement** in a header comment.
3.  **Implement Fix:** Code the solution.
4.  **One-Click Verification:** Run `npm run verify` to execute the full pipeline:
    - `proto:generate`: Refresh Protobuf contracts.
    - `lint:fix`: Auto-fix linting issues.
    - `test`: Run all unit and integration tests.
    - `build`: Perform type-checking (`tsc`) and production build.
    - `format`: Finalize code formatting.
6.  **Debugging Tests:** To debug tests using the Gemini CLI `node-debugger` tool:
    - **Step 1:** Start the test in the background using `run_shell_command`:
      `node node_modules/vitest/vitest.mjs --inspect-brk=9229 --no-file-parallelism run <path_to_test> &`
    - **Step 2:** Attach to the worker's debug port using `attach_debugger` (usually port `9229` as specified in the command).
    *Note: Using `start_node_process` on the CLI directly causes a "double-pause" (once for the CLI, once for the worker). Using the shell to start the worker directly is more efficient.*
6.  **Commit:** Ensure the test remains as a permanent artifact to prevent future regressions.

---
