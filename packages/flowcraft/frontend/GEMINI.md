# Flowcraft Project Context

Flowcraft is a high-performance, backend-driven node-based editor built with React and `@xyflow/react` (React Flow 12). It features a unified incremental update protocol using Protobuf and a robust task system for long-running operations.

## Architecture & Core Technologies

- **Frontend:** React 19, Vite, TypeScript.
- **Graph Engine:** `@xyflow/react` for node and edge management.
- **State Management:**
  - **Global:** Zustand.
  - **Sync & Persistence:** **SpacetimeDB** is the Source of Truth for the graph (Nodes, Edges, Viewport) and Task status.
  - **Undo/Redo:** `zundo` (Zustand middleware) for local history management.
- **Layout:** `dagre` for automatic directed graph positioning.
- **Protocol:** Protocol Buffers (Protobuf) for data serialization.
- **Backend (Decentralized):**
  - **SpacetimeDB (Port 3000):** Real-time synchronization and state management.
  - **Node.js Worker/Server (Port 3001):**
    - **Worker**: Subscribes to SpacetimeDB `tasks` table to execute AI actions and node operations.
    - **Metadata Provider**: Handles legacy `TemplateDiscovery` and `InferenceConfig` via ConnectRPC.
    - **Asset Server**: Manages file uploads and static assets.

## Key Systems

### 1. Node Architecture

Nodes are dynamic and driven by backend-defined JSON/Protobuf schemas.

- **DynamicNode**: A modular component that can switch between **Media Mode** (images, video, markdown) and **Widgets Mode** (interactive fields like sliders, selects, etc.).
- **ProcessingNode**: A temporary placeholder for ongoing tasks, providing live progress feedback and cancellation.
- **Port System**: Supports explicit and implicit ports. Implicit ports are tied to widgets and can toggle a widget's enabled state when connected.

### 2. Protocol & Communication

- **Hybrid Synchronization**:
  - **Graph State**: Handled by **SpacetimeDB** (Nodes, Edges, Viewport). Changes are pushed via Reducers and pulled via Subscriptions.
  - **Legacy/Tasks**: **FlowMessage Envelope** (Protobuf) via ConnectRPC/gRPC. Used for `actionExecute`, `taskUpdate`, and `templateDiscovery`.
- **Incremental Mutations**: The graph state is updated via atomic mutations (`addNode`, `updateNode`, etc.).
  - **SpacetimeDB**: Mutations are mapped to SpacetimeDB Reducers.
  - **ConnectRPC**: (Deprecated for Graph) Still used for some specialized updates.
- **RJSF & JSON Schema Integration**:
  - **Contract-First Forms**: UI forms for node/action configurations are driven by `react-jsonschema-form` (RJSF).
  - **Schema Generation**: JSON Schemas are generated from Protobuf definitions. There is no hard requirement for the file location, but targets must be manually added to `buf.gen.schema.yaml`.
  - **Keep Artifacts Clean**: Only include messages intended for user-facing configuration (RJSF) in the schema generation process. NEVER run the `jsonschema` generator on entire core or state packages. This ensures the frontend bundle stays lean and the schema registry remains free of redundant or internal state schemas.
- **Hydration**: Client-side handlers are re-attached to server-provided node data using a centralized hydration utility.
- **Client**: `socketClient` (Connect/gRPC) handles all communication with the Node.js backend.
- **Hooks**: `useFlowSocket` provides the main interface for components to interact with the backend.

### 3. Task & Job System

- Long-running operations (e.g., AI generation) use a task store to track lifecycles (`pending`, `processing`, `completed`, `failed`).
- Real-time updates are streamed from the mock backend to the frontend.

## Building and Running

- **Development:** `npm run dev` starts all components concurrently:
  - **Vite (Port 5173)**: Frontend HMR.
  - **SpacetimeDB (Port 3000)**: Backend database.
  - **Node.js (Port 3001)**: Worker and Metadata provider.
- **Build:** `npm run build`
- **Linting:** `npm run lint`
- **Formatting:** `npm run format`
- **Protobuf Generation:** `npm run proto:generate`
- **Spacetime Bindings:** `npm run spacetime:generate`

## Development Conventions

- **Typing:** Strict TypeScript typing is enforced. Avoid `any` in favor of `unknown`, generics, or generated Protobuf types. **Prefer Enums over string literals or string unions wherever possible** to ensure type safety, discoverability, and centralized definition.
- **Event Bus:** Use the centralized event bus in `flowStore` (`dispatchNodeEvent`) for inter-component signaling (e.g., opening previews or editors). **Define event names as Enums.**
- **Theming:** The application is dark-mode first. Use CSS variables defined in `index.css` (`--node-bg`, `--primary-color`, etc.) for all styling.
- **Components & Performance:**
  - Prefer Functional Components and Hooks.
  - **Memoization:** Proactively wrap node renderers and complex widgets in `React.memo`. Always use `useCallback` for event handlers (especially those passed to handles/resizers) to prevent unnecessary React Flow re-renders during dragging.
  - **Error Resilience:** Wrap complex node renderers in local Error Boundaries. A failure in a specific node's media or widget renderer must not crash the entire editor canvas.
- **Data & Synchronization:**
  - **Defensive Parsing:** When parsing backend-provided JSON (e.g., `widgetsSchemaJson`, `widgetsValues`), always use `try-catch` blocks and provide sensible fallback/default values.
  - **Mutation Atomicity:** All persistent graph changes (position, data, edges) MUST go through `applyMutations` to ensure synchronization with SpacetimeDB. Avoid mixing local component state with store state for data that needs to be synchronized.
- **Modularity & File Size:** Keep components small and focused. **A single file should ideally not exceed 300 lines of code.**
  - **Exemption:** Components in `src/components/ui/` and `src/components/ai-elements/` (typically introduced via shadcn or external libraries) are exempt from this line limit and should not be refactored solely to satisfy this rule.
- **Imports & Path Aliases:** Use the `@/` alias for all absolute imports from the `src` directory (e.g., `import { useFlowStore } from "@/store/flowStore"`). Avoid deep relative paths (e.g., `../../hooks/...`) whenever possible to improve maintainability and readability.
- **Structural Refactoring (ast-grep):** Use `npx ast-grep` for precise, syntax-aware code transformations across the project. 
  - *Example:* `npx ast-grep --pattern '$O.reducers.$M($$A)' --rewrite '$O.pbreducers.$M($$A)'` helps migrate legacy calls.
  - Prefer AST-based tools over simple regex/sed for complex nested structures to avoid breaking code context.
- **Polymorphism over Conditionals:**
  - **Avoid `if/else` or `switch` on node types** in shared logic (hooks, stores, utils).
  - Use **Data-Driven Design**: Define type-specific constraints (min-size, default data) in `NodeTemplate` or dedicated config registries (e.g., `mediaConfigs.ts`).
  - Presentation fields like `position`, `width`, `height`, and `parentId` must be **strictly typed** in Protobuf and handled as "pass-through" by the backend to ensure visual consistency while allowing introspection.

## Advanced Bridging & Communication

### 1. Dual-Track Reducers (PB + Native)
The system supports two parallel ways to call SpacetimeDB reducers:
- **`conn.reducers`**: Original SpacetimeDB accessors. Expects `Uint8Array` for Protobuf-backed fields.
- **`conn.pbreducers`**: Enhanced accessors. Transparently accepts standard Protobuf Message objects and performs serialization automatically.
- **Type Safety**: The type of `pbreducers` is automatically projected from the native definition via `PbReducersProjection`. NEVER manually define `interface PbReducers`.

### 2. Automatic Metadata Discovery
The generation pipeline (`scripts/generate-pb-client.ts`) automatically discovers mappings between STDB Tables/Reducers and Protobuf Schemas by:
- Mocking the STDB server environment and capturing table definitions.
- Heuristically matching TypeNames (e.g., `Node` in STDB maps to `NodeSchema` in PB).
- Generating a pure metadata file `src/generated/pb_metadata.ts`.

### 3. Data Normalization
Always use `convertStdbToPb` (from `@/utils/pb-client`) to ingest data from SpacetimeDB subscriptions. 
- **Reason**: SpacetimeDB may represent Enums as objects (e.g., `{ mode: 1 }`). This utility "washes" the data back into standard Protobuf primitive format before it hits the store.
- **Consistency**: Use the bridge at the entry point (`useSpacetimeSync`) so that the rest of the application (UI components and stores) only deals with standardized Protobuf shapes.

## Quality & Development Workflow

To maintain high code quality and prevent regressions, follow this iterative process for every feature or bug fix:

1.  **Analyze & Reproduce:** Understand the requirement or bug.
2.  **Write Tests First (or during):** Create a test file in `__tests__/` that explicitly describes the **Problem** and the **Requirement** in a header comment.
3.  **Implement Fix:** Code the solution.
4.  **One-Click Verification:** Run `npm run verify` to execute the full pipeline:
    - `lint:fix`: Auto-fix linting issues (Excludes `src/components/ui` and `src/components/ai-elements` from mandatory manual refactoring).
    - `test`: Run all unit and integration tests.
    - `build`: Perform type-checking (`tsc`) and production build.
    - `format`: Finalize code formatting.
5.  **Debugging Tests:** To debug tests using the Gemini CLI `node-debugger` tool:
    - **Step 1:** Start the test in the background using `run_shell_command`:
      `node node_modules/vitest/vitest.mjs --inspect-brk=9229 --no-file-parallelism run <path_to_test> &`
    - **Step 2:** Attach to the worker's debug port using `attach_debugger` (usually port `9229` as specified in the command).
      _Note: Using `start_node_process` on the CLI directly causes a "double-pause" (once for the CLI, once for the worker). Using the shell to start the worker directly is more efficient._
6.  **Commit:** Ensure the test remains as a permanent artifact to prevent future regressions.
