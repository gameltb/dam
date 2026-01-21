# Distributed NodeKernel Design Draft

## 1. Overview
The **Distributed NodeKernel** is a unified execution layer designed to bridge the gap between frontend interactions and multi-environment/multi-language background workers. It treats both the Browser and remote Workers as execution nodes that subscribe to a shared task queue managed by **SpacetimeDB**.

## 2. Core Components

### 2.1 Task Orchestrator (SpacetimeDB)
The central authority for task lifecycle. It ensures atomic operations for claiming tasks and prevents node-level concurrency conflicts.
- **`tasks` Table**: Tracks `id`, `nodeId`, `taskType`, `status`, `ownerId`, `paramsPayload`, and `result`.
- **`workers` Table**: Registry of active execution instances with their `capabilities`, `lang`, and `lastHeartbeat`.
- **`task_audit_log` Table**: Durable log of task events (errors, milestones).
- **Reducers**: Atomic logic for `submitTask`, `claimTask`, `updateTaskProgress`, `completeTask`, and `failTask`.

### 2.2 NodeKernel (Shared Logic)
A library implemented in TypeScript (and portable to Python) that provides:
- **Busy Guard**: Pre-execution check (`checkNodeBusy`) to ensure a node isn't already running a task.
- **TaskContext Creator**: Generates a standardized `TaskContext` for logic execution.
- **ORM-style Drafting**: `nodeDraft` helper allowing workers to update node state using a safe proxy pattern.

### 2.3 Execution Environment (Workers)
- **Browser Worker**: (Planned) Local inference or UI-heavy coordination.
- **Node.js Worker**: Current default worker handling OpenAI integration and system tasks.
- **Python Worker**: (Future) Specialized for heavy ML models (SD, Torch).

---

## 3. Detailed Implementation Details

### 3.1 Unified Task Execution (`TaskContext`)
The `TaskContext` provides a consistent API for business logic regardless of where it runs:
- `updateProgress(percentage, message)`: Real-time UI feedback.
- `log(message, level)`: Detailed audit logging in the DB.
- `complete(result)`: Finalizes the task and unlocks the node.
- `fail(error)`: Records the failure and notifies the UI.
- `isCancelled()`: Checks if the user requested a termination.

### 3.2 Atomic Task Claiming
To support multiple worker instances, the `claimTask` reducer ensures safety:
1. Worker receives a `PENDING` task via subscription.
2. Worker calls `claimTask(taskId, workerId)`.
3. STDB checks if `status == PENDING` atomically.
4. If successful, `status` becomes `CLAIMED` and `ownerId` is locked.
5. Other workers attempting to claim the same task will receive a `TASK_ALREADY_CLAIMED` error.

### 3.3 Worker Self-Registration & Heartbeats
Workers automatically register on startup via `registerWorker`:
- **Capabilities**: A list of task types they can handle (e.g., `["chat.openai", "image.gen"]`).
- **Heartbeat**: Every 5 seconds, the worker updates its `lastHeartbeat` in STDB.
- **Ghost Detection**: The system can identify crashed workers by checking for stale heartbeats and automatically failing or re-queueing their tasks.

---

## 4. Application Examples

### 4.1 Chat Generation Logic
1. **Frontend**: Calls `submitTask(nodeId, "chat.openai", { model, prompt })`.
2. **Worker**: 
   - Claims the task.
   - Updates status to `RUNNING`.
   - Streams chunks from OpenAI.
   - Calls `conn.pbreducers.updateChatStream` for real-time "typing" effect.
   - Calls `ctx.complete({ messageId })` when finished.
3. **UI**: Observes `tasks` table and `chat_streams` table to render progress and text.

### 4.2 Image Generation Logic (Multi-Worker)
1. **Frontend**: Submits task with type `image.gen`.
2. **Cluster**: Multiple Python workers on different GPUs see the task.
3. **Winner**: One worker claims it, downloads the prompt, and runs the diffusion model.
4. **Progress**: Worker calls `ctx.updateProgress` for every sampling step.
5. **Result**: Worker uploads image to asset server, then calls `ctx.complete({ url })`.
6. **Persistence**: Kernel uses `nodeDraft` to automatically update the node's `VisualNodeState` with the new URL.

---

## 5. Robustness & Self-Healing
- **Disconnected Clients**: If a frontend client disconnects, `spacetimedb.clientDisconnected` cleans up transient assignments.
- **Durable Progress**: Since progress is in the DB, refreshing the browser does not lose the current task's state; the UI will simply re-attach to the existing `NodeRuntimeState`.
- **Pre-emptive Failure**: UI can hide "Generate" buttons if no workers with the required capability are currently in the `workers` table.

## 6. Current Schema (Protobuf)

```protobuf
enum TaskStatus {
  TASK_STATUS_PENDING = 0;
  TASK_STATUS_CLAIMED = 1;
  TASK_STATUS_RUNNING = 2;
  TASK_STATUS_COMPLETED = 3;
  TASK_STATUS_FAILED = 4;
  TASK_STATUS_CANCELLED = 5;
}

message TaskDefinition {
  string task_id = 1;
  string node_id = 2;
  string task_type = 3;
  bytes params_payload = 4;
  WorkerSelector selector = 5;
  int64 created_at = 6;
}

message WorkerInfo {
  string worker_id = 1;
  WorkerLanguage lang = 2;
  repeated string capabilities = 3;
  map<string, string> tags = 4;
  int64 last_heartbeat = 5;
}
```