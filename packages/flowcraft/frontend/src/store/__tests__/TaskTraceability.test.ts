import { describe, it, expect, beforeEach } from "vitest";
import { useTaskStore } from "../taskStore";
import { useFlowStore } from "../flowStore";
import { MutationSource, TaskStatus } from "../../types";
import { GraphMutationSchema } from "../../generated/flowcraft/v1/core/service_pb";
import { create } from "@bufbuild/protobuf";

import { initStoreOrchestrator } from "../orchestrator";
import { beforeAll } from "vitest";

describe("Task Traceability", () => {
  beforeAll(() => {
    initStoreOrchestrator();
  });

  beforeEach(() => {
    useFlowStore.getState().resetStore();
    useTaskStore.getState().resetStore();
  });

  it("should correlate mutations with tasks in the log", () => {
    const taskId = "task-1";
    const taskStore = useTaskStore.getState();
    const flowStore = useFlowStore.getState();

    // 1. Register a task
    taskStore.registerTask({
      taskId,
      label: "Background Generation",
      source: MutationSource.SOURCE_REMOTE_TASK,
    });

    // 2. Apply mutations with task context
    const mutations = [
      create(GraphMutationSchema, {
        operation: {
          case: "clearGraph",
          value: {},
        },
      }),
    ];

    flowStore.applyMutations(mutations, {
      taskId,
      source: MutationSource.SOURCE_REMOTE_TASK,
      description: "AI result applied",
    });

    // 3. Verify correlation
    const logs = useTaskStore.getState().mutationLogs;
    expect(logs.length).toBe(1);
    expect(logs[0]?.taskId).toBe(taskId);
    expect(logs[0]?.source).toBe(MutationSource.SOURCE_REMOTE_TASK);

    const task = useTaskStore.getState().tasks[taskId];
    expect(task?.mutationIds.length).toBe(1);
    expect(task?.mutationIds[0]).toBe(logs[0]?.id);
  });

  it("should record user-initiated actions as anonymous logs if no taskId", () => {
    const flowStore = useFlowStore.getState();

    flowStore.applyMutations(
      [
        create(GraphMutationSchema, {
          operation: { case: "clearGraph", value: {} },
        }),
      ],
      { description: "User cleared canvas" },
    );

    const logs = useTaskStore.getState().mutationLogs;
    expect(logs.length).toBe(1);
    expect(logs[0]?.taskId).toBe("manual-action");
    expect(logs[0]?.source).toBe(MutationSource.SOURCE_USER);

    const task = useTaskStore.getState().tasks["manual-action"];
    expect(task?.label).toContain("Manual");
  });

  it("should update task status via taskStore", () => {
    const taskId = "task-status-1";
    const store = useTaskStore.getState();

    store.registerTask({ taskId, label: "Test" });
    store.updateTask(taskId, {
      status: TaskStatus.TASK_COMPLETED,
      progress: 100,
    });

    expect(useTaskStore.getState().tasks[taskId]?.status).toBe(
      TaskStatus.TASK_COMPLETED,
    );
  });
});
