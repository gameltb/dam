# Omni-Graph Architecture: Entity-Component-Protocol (ECP)

## 1. Vision
The Omni-Graph architecture transforms Flowcraft from a simple node editor into a distributed, reactive execution environment. It decouples **what a node is** (Entity), **what data it holds** (Components), and **how it communicates** (Protocols).

## 2. Core Concepts

### A. Entity (The ID)
A node is just a unique ID in a Scope. It has no hardcoded logic.
- `nodeId`: UUID
- `templateId`: References a behavior template.
- `scopeId`: Logical container (allows nesting).

### B. Components (The Data)
Data is split into specialized tables in SpacetimeDB. A node "assembles" its state by attaching components.
- **TransformComponent**: `[x, y, width, height]` (UI only).
- **StateComponent**: `[pb_payload]` (Business logic state, Protobuf).
- **MetadataComponent**: `[displayName, tags]` (Opaque info).
- **ProxyComponent**: `[sourceNodeId]` (Optional: makes this a "Shadow Node").

### C. Protocols (The Ports)
Ports are **Protocol Endpoints**. They don't just pass strings; they negotiate contracts.
- **Protocol Family**: e.g., `flowcraft.v1.streaming`, `flowcraft.v1.tensor`.
- **Contract**: Defines valid message types for both **Downstream** (Data) and **Upstream** (Signals).
- **Bi-directional Signaling**: Input ports can send "Back-pressure" or "Cancellation" signals to Output ports.

---

## 3. Data Schema (Logical)

### Nodes & Components
```sql
TABLE nodes (nodeId, templateId, scopeId);
TABLE node_transforms (nodeId, x, y, w, h);
TABLE node_data (nodeId, state_pb); -- The Protobuf blob
TABLE node_proxies (nodeId, targetNodeId); -- Shadow node mapping
```

### Edges as Channels
```sql
TABLE edges (edgeId, sourceNodeId, sourcePortId, targetNodeId, targetPortId, protocolFamily);
```

### Signals (The Pulse)
```sql
TABLE node_signals (
    signalId, 
    sourceNodeId, 
    targetPortId, 
    payload_pb, -- Signal type (Stop, Recompute, Flush)
    timestamp
);
```

---

## 4. Execution Model: The Declarative State Machine

Instead of a linear "Run" command, the graph acts as a **Perpetual State Machine**.

1.  **Observation**: Workers subscribe to `node_data` and `edges`.
2.  **Reactive Trigger**: When a node's state changes, or a signal arrives at a port, the Worker evaluates the node's **Behavior Function**.
3.  **State Transition**: The Worker updates the node's `StateComponent` (e.g., changing status from `PENDING` to `PROCESSING`).
4.  **Propagation**: The Worker looks up connected `edges` and dispatches data/signals to target entities.

---

## 5. Advanced Features

### Shadow Nodes (Proxies)
- A node in Scope A can be "shadowed" into Scope B.
- The shadow node has its own `TransformComponent` (so it can be positioned differently) but shares the `StateComponent` of the original.
- Changes to the original reflect instantly in all shadows.

### Nested Scopes (Closures)
- A Group node can act as a **Closure**.
- It encapsulates a sub-graph.
- Boundary ports on the Group node map internal sub-graph protocols to external parent-graph protocols.

---

## 6. Implementation Strategy (The "State Machine First" Path)

### Phase 1: Infrastructure & Componentization
- [ ] Refactor `AppNode` to strictly separate `presentation` from `data`.
- [ ] Migrate `nodeMaterializer` to handle component-wise updates.
- [ ] Implement `sanitizeNode` as the central entity factory.

### Phase 2: Declarative State Machine (Target)
- [ ] Define a standard `NodeStatus` enum in Protobuf (`Idle`, `Pending`, `Running`, `Error`, `Completed`).
- [ ] Update Workers to react to `StateComponent` transitions.
- [ ] Implement "Trigger-on-Change" logic for all core nodes.

### Phase 3: Protocol-based Ports
- [ ] Add `protocolFamily` to `Port` definition in Protobuf.
- [ ] Implement UI-level visual encoding for different protocols (color-coded ports).
- [ ] Create the `node_signals` table for upstream feedback.

### Phase 4: Proxies & Scopes
- [ ] Implement `node_proxies` logic in `GraphMapper`.
- [ ] Add "Create Shadow" to the context menu.
- [ ] Enable cross-scope reactive updates.
