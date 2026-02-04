# Protobuf Design & Implementation Guidelines (V3.0)

This document defines the integrated design strategy for Protobuf messages, SpacetimeDB table structures, and Frontend state management in the Flowcraft project.

## 1. Layering Strategy

All business objects are divided into three strictly isolated logical layers. Do not mix fields across these layers:

| Layer            | Responsibility                     | Examples                    | Storage Implementation                                   |
| :--------------- | :--------------------------------- | :-------------------------- | :------------------------------------------------------- |
| **Identity**     | Unique ID, relationships, indexing | `node_id`, `parent_id`      | SpacetimeDB Primary Key / Indexed Columns (Strings)      |
| **Presentation** | Visual state (Position, Size)      | `x`, `y`, `width`, `height` | Flattened columns in `node_transforms` / `node_metadata` |
| **Domain**       | Business logic & extension state   | `tree_id`, `extension`      | `byteArray` in SpacetimeDB with `__pb_schema` hints      |

## 2. Naming & Semantics

Use suffixes to clarify the lifecycle and intent of a message:

- **`...State`**: Persistent storage objects. Represents "Fact". Stored in DB `byteArray` columns.
- **`...Params`**: Command parameters. Represents "Intent". Used as input for Reducers or Action execution.
- **`...Request`**: High-level RPC/Service commands.
- **`...Response`**: Synchronous feedback from a service.
- **`...Event`**: Asynchronous notification. Used for streaming or ephemeral signaling.

## 3. Polymorphism with `oneof`

To handle diverse node types efficiently, use the `oneof` pattern in `NodeData`:

```proto
message NodeData {
  // ... common fields
  oneof extension {
    ChatNodeState chat = 51;
    AiGenNodeState ai_gen = 52;
    VisualNodeState visual = 53;
  }
}
```

**Guideline**: Always include an `extension` in `NodeData` for specific business logic. In the Frontend, this enables **Discriminated Union** narrowing for perfect type inference.

## 4. UI Rendering (RJSF) Integration

For messages used to generate UI via `react-jsonschema-form`:

- **Flattened Design**: Keep configuration fields shallow. Nesting beyond 3 levels degrades UI usability.
- **Strong Typing**: Avoid `google.protobuf.Struct` where possible. Use explicit types in `oneof` branches.
- **Enum Safety**: Every Enum MUST have an `_UNSPECIFIED = 0` value to prevent RJSF from defaulting to a valid but unintended business option.

## 5. Organization & Packages

- **Semantic Paths**: `core/` (foundation), `nodes/` (specific state), `actions/` (execution), `services/` (communication).
- **Package Matching**: `package flowcraft.v1.[module];` must exactly match the physical directory structure.
- **Absolute Imports**: Always import from the root: `import "flowcraft/v1/core/base.proto";`.

## 6. Implementation Standards

### 6.1 Strict Hydration

When ingesting data from SpacetimeDB, always use Buf's `create(Schema, data)` utility.

- **Why**: Ensures that repeated fields (arrays) and maps are initialized correctly, avoiding runtime "undefined" errors.

### 6.2 Standardized Serialization

- Use `toJsonString(Schema, message)` for persisting PB data into string-based columns or logs.
- Use `fromJsonString(Schema, json)` for parsing.
- Avoid raw `JSON.stringify` on Protobuf objects as it fails to handle `int64` (BigInt) and specific field naming conventions correctly.

### 6.3 State Mutation Patterns

- **Low-level changes** (Position, Name, simple data fields): Use the **Immer Patch Pipeline** via `applyMutations(recipe)`.
- **High-level commands** (Execute AI, Subgraph operations): Use specialized Reducers/RPCs with `...Request` messages.

## 7. Quality Metrics

- **Zero `any`**: All message handling must be strictly typed via generated TS files.
- **English Only**: All comments, field names, and documentation must be in English.
- **Fail-Fast**: If a message fails schema validation or mapping, throw an explicit `Error`. Never allow silent fallbacks.
