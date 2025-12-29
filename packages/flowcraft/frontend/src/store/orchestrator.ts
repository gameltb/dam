import { MutationSource } from "../types";
import {
  type GraphMutation,
  GraphMutationSchema,
} from "../generated/core/service_pb";
import { create } from "@bufbuild/protobuf";
import { useFlowStore } from "./flowStore";
import { useTaskStore } from "./taskStore";

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
        mutations: GraphMutation[];
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
        mutations: mutations.map((m) => create(GraphMutationSchema, m)),
      });
    }
  });
}
