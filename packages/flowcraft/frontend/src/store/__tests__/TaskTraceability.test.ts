import { describe, it, expect, beforeEach } from "vitest";
import { useFlowStore } from "../flowStore";
import { useTaskStore } from "../taskStore";
import { MutationSource } from "../../types";

/**
 * PROBLEM: Changes to the graph were hard to trace back to their origin (User vs Task).
 * REQUIREMENT: Every mutation should be associated with a taskId and logged in the TaskStore.
 */
describe("Task-based Mutation Traceability", () => {
  beforeEach(() => {
    // Reset stores
    useTaskStore.setState({ tasks: {}, mutationLogs: [] });
    useFlowStore.setState({ nodes: [], edges: [], version: 0 });
  });

  it("should automatically log mutations to the task store when applyMutations is called", () => {
    const taskId = "test-task-123";
    const mutations = [{ clearGraph: {} }];

    // 1. Apply mutations with context
    useFlowStore.getState().applyMutations(mutations, {
      taskId,
      source: MutationSource.REMOTE_TASK,
      description: "Remote cleanup",
    });

    // 2. Verify task was registered
    const task = useTaskStore.getState().tasks[taskId];
    expect(task).toBeDefined();
    expect(task.source).toBe(MutationSource.REMOTE_TASK);
    expect(task.label).toBe("Remote cleanup");

    // 3. Verify mutation log exists and is linked
    const logs = useTaskStore.getState().mutationLogs;
    expect(logs.length).toBe(1);
    expect(logs[0].taskId).toBe(taskId);
    expect(logs[0].mutations).toEqual(mutations);
    expect(task.mutationIds).toContain(logs[0].id);
  });

  it("should use a default task if no context is provided", () => {
    useFlowStore.getState().applyMutations([{ clearGraph: {} }]);

    const defaultTaskId = "manual-interaction";
    const task = useTaskStore.getState().tasks[defaultTaskId];

    expect(task).toBeDefined();
    expect(task.source).toBe(MutationSource.USER);
  });
});
