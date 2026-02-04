import { useFlowStore } from "../flowStore";
import { type GraphMiddleware, MutationDirection } from "./types";

/**
 * HistoryMiddleware
 * Responsible for recording changes to the Undo stack.
 * Note: To simplify the implementation, we still use full snapshot mode, but it's triggered via the pipeline.
 */
export const historyMiddleware: GraphMiddleware = (event, next) => {
  const { context, direction } = event;

  // 1. Only record changes that are locally emitted and not history operations themselves
  if (direction === MutationDirection.OUTGOING && !context.isHistoryOp) {
    const store = useFlowStore.getState();

    // Certain lightweight changes (e.g., selection) might not need to enter the history stack
    const shouldSnapshot = event.patches.some(
      (p) =>
        !p.path.includes("selected") && (!p.path.includes("position") || !context.description?.includes("dragging")),
    );

    if (shouldSnapshot) {
      store.takeSnapshot();
    }
  }

  next(event);
};
