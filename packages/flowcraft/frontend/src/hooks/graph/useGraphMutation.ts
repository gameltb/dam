import { useCallback } from "react";

import { commit } from "@/store/orchestrator";

/**
 * useGraphMutation
 */
export const useGraphMutation = () => {
  const updateViewport = useCallback((x: number, y: number, zoom: number) => {
    commit(
      (draft) => {
        draft.viewport = { x, y, zoom };
      },
      { description: "Update viewport", transient: true },
    );
  }, []);

  return { updateViewport };
};
