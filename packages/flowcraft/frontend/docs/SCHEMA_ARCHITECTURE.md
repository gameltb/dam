# Protocol-Bridge Architecture (V3.0)

This document details the Flowcraft synchronization engine, which bridges SpacetimeDB's relational storage with the Frontend's hierarchical Protobuf-driven domain model.

## 1. Core Architecture

### 1.1 Protobuf as Domain Model

The system uses Protobuf v2 definitions (`schema/flowcraft/v1/`) as the single source of truth for the **Domain Model**.

- **Frontend**: The Zustand store (`flowStore.ts`) holds state in standardized Protobuf shapes.
- **Backend**: SpacetimeDB logic and Workers process binary Protobuf streams.

### 1.2 Binary-First Storage with Metadata Hints

Instead of direct relational mapping for complex types, Flowcraft uses a **Binary-first approach** in SpacetimeDB to maintain high performance and schema flexibility.

- **Columns**: Complex objects are stored as `byteArray` in SpacetimeDB.
- **Hints**: Table definitions in `spacetime-module/src/tables/` use the `__pb_schema` hint to tell the build pipeline which Protobuf Message maps to which column.

```typescript
// Example: spacetime-module/src/tables/core.ts
export const nodeData = table(
  { name: "node_data", public: true },
  {
    nodeId: t.string().primaryKey(),
    state: Object.assign(t.byteArray(), { __pb_schema: "NodeData" }),
  },
);
```

---

## 2. Automated Bridge Pipeline

The mapping between relational rows and domain objects is automated via `scripts/generate-pb-client.ts`.

### 2.1 Metadata Discovery

The script mocks the SpacetimeDB environment, captures table schemas, and searches for `__pb_schema` hints. It generates `src/generated/pb_metadata.ts`, which contains:

- **`PB_REDUCERS_MAP`**: Tells the client which Reducer arguments need automatic Protobuf serialization.
- **`TABLE_TO_PROTO`**: Tells the client which Table columns need automatic Protobuf deserialization.

### 2.2 Dual-Track Reducers

The `PbConnection` wrapper provides two ways to interact with the backend:

- **`conn.reducers`**: Native STDB access (requires `Uint8Array`).
- **`conn.pbreducers`**: Enhanced access (accepts plain JS objects, performs auto-serialization).

---

## 3. Synchronization & Reconciliation

### 3.1 Outgoing: Patch-Based Mutations

Flowcraft uses **Immer Patches** to drive outgoing changes.

1.  **Recipe**: Components provide a "recipe" to `applyMutations`.
2.  **Patches**: Immer generates fine-grained patches.
3.  **Sync Middleware**: Iterates through patches and calls the specific `pbreducers` mapped to the modified paths.

### 3.2 Incoming: Mirror Repository Reconciliation

Remote changes are merged using the **Mirror Repository pattern** in `useGraphSync.ts`:

1.  **Mapping**: `GraphMapper` aggregates flattened STDB rows (Transform, Metadata, Data) into a single domain `AppNode`.
2.  **Indexing**: Tables are pre-indexed into Maps for O(1) reconciliation performance.
3.  **Hydration**: Every incoming object is forced through Buf's `create(Schema, data)` to ensure all default values (like empty arrays) are populated, preventing runtime `undefined` errors.

---

## 4. Development Quality

- **English Comments**: All source code and documentation must use English.
- **Strict Typing**: No `as any` allowed in mapping or business logic.
- **Serialization**: Use Buf's `toJsonString` and `fromJsonString` for persisting PB messages into string columns (e.g., `parts_json`).
