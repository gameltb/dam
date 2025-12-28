import { useFlowStore } from "./flowStore";
import { useTaskStore } from "./taskStore";
import { MutationSource } from "../types";
import { flowcraft_proto } from "../generated/flowcraft_proto";

/**
 * Orchestrates side-effects between different stores.
 * This decouples flowStore from taskStore.
 */
export function initOrchestrator() {
  // Listen to node events for mutation logging
  useFlowStore.subscribe((state, prevState) => {
    if (
      state.lastNodeEvent !== prevState.lastNodeEvent &&
      state.lastNodeEvent?.type === "mutations-applied"
    ) {
      const { mutations, context } = state.lastNodeEvent.payload as {
        mutations: flowcraft_proto.v1.IGraphMutation[];
        context?: {
          taskId?: string;
          source?: MutationSource;
          description?: string;
        };
      };

      const taskId = context?.taskId ?? "manual-interaction";
      const taskStore = useTaskStore.getState();

      if (!taskStore.tasks[taskId]) {
        taskStore.registerTask({
          taskId,
          label: context?.description ?? "Action",
          source: context?.source ?? MutationSource.USER,
        });
      }

      taskStore.logMutation({
        taskId,
        source: context?.source ?? MutationSource.USER,
        description: context?.description ?? "Applied",
        mutations,
      });
    }
  });
}
