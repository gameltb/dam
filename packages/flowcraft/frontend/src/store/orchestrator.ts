import { applyPatches, produceWithPatches } from "immer";

import { type AppNode, type MutationContext } from "@/types";

import { useFlowStore } from "./flowStore";
import { historyMiddleware } from "./middleware/historyMiddleware";
import { pipeline } from "./middleware/pipeline";
import { syncMiddleware } from "./middleware/syncMiddleware";
import { taskMiddleware } from "./middleware/taskMiddleware";
import { MutationDirection } from "./middleware/types";
import { type RFState } from "./types";

/**
 * Orchestrator.commit
 *
 * The single entry point for all state mutations in the project.
 * Replaces the previously scattered 'applyMutations' calls.
 */
export const commit = (recipe: (draft: RFState) => void, context: MutationContext = {}) => {
  const store = useFlowStore.getState();

  // 1. Generate patches (Immer)
  const [_, patches, inversePatches] = produceWithPatches(store, recipe);
  if (patches.length === 0) return;

  // 2. Execute pipeline (History -> Task -> Sync)
  pipeline.execute(
    {
      context,
      direction: MutationDirection.OUTGOING,
      inversePatches,
      patches,
    },
    (event) => {
      // 3. Apply patches to Zustand
      const nextState = applyPatches(useFlowStore.getState(), event.patches);
      useFlowStore.setState(nextState);

      // 4. Handle view update logic (if structural changes occurred)
      const hasStructuralChanges = event.patches.some(
        (p) => p.path.includes("nodesById") || p.path.includes("edgesById"),
      );
      if (hasStructuralChanges || context.isInteractionEnd) {
        useFlowStore.getState().refreshView();
      }
    },
  );
};

/**
 * Orchestrator.editNode
 */
export const editNode = (nodeId: string, recipe: (node: AppNode) => void) => {
  commit(
    (draft) => {
      const node = draft.nodesById[nodeId];
      if (node) recipe(node);
    },
    { description: `Edit node ${nodeId}` },
  );
};

/**
 * Orchestrator initialization: Registering core middlewares.
 */
export const initStoreOrchestrator = () => {
  pipeline.clear().use(historyMiddleware).use(taskMiddleware).use(syncMiddleware);

  console.log("[Orchestrator] Initialized with History, Task, and Sync middlewares");
};
