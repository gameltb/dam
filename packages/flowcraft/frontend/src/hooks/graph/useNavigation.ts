import { useReactFlow } from "@xyflow/react";
import { useCallback } from "react";

import { useFlowStore } from "@/store/flowStore";
import { NavigationStatus, useNavigationStore } from "@/store/ui/navigationStore";

/**
 * useNavigation
 * Centralized navigation controller. Responsible for managing camera animations, state synchronization, and viewport recovery during scope switching.
 */
export function useNavigation() {
  const { fitView, setViewport } = useReactFlow();
  const setActiveScope = useNavigationStore((s) => s.setActiveScope);
  const setStatus = useNavigationStore((s) => s.setNavigationStatus);
  const getViewportForScope = useNavigationStore((s) => s.getViewportForScope);
  const nodesById = useFlowStore((s) => s.nodesById);

  /**
   * Navigate to a specified scope
   */
  const navigateTo = useCallback(
    async (targetScopeId: null | string, options: { focusNodeId?: string; immediate?: boolean } = {}) => {
      const currentStatus = useNavigationStore.getState().navigationStatus;
      if (currentStatus !== NavigationStatus.IDLE) return;

      // 1. Start transition: camera focus or blur
      setStatus(NavigationStatus.TRANSITIONING);

      if (!options.immediate) {
        if (options.focusNodeId) {
          await fitView({ duration: 400, nodes: [{ id: options.focusNodeId }] });
        }
        // Give a little visual dwell time
        await new Promise((resolve) => setTimeout(resolve, 50));
      }

      // 2. Swap scope (Swapping)
      setStatus(NavigationStatus.SWAPPING);
      setActiveScope(targetScopeId);

      // 3. Restore viewport
      const savedViewport = getViewportForScope(targetScopeId);
      if (savedViewport) {
        setViewport(savedViewport, { duration: 0 });
      } else {
        fitView({ duration: 0 });
      }

      // 4. Finish transition
      setTimeout(() => {
        setStatus(NavigationStatus.IDLE);
      }, 50);
    },
    [fitView, setViewport, setActiveScope, setStatus, getViewportForScope],
  );

  /**
   * Return to the previous scope level
   */
  const goBack = useCallback(() => {
    const { activeScopeId } = useNavigationStore.getState();
    if (!activeScopeId) return;

    const currentNode = nodesById[activeScopeId];
    const parentScopeId = currentNode?.scopeId || null;

    navigateTo(parentScopeId);
  }, [nodesById, navigateTo]);

  return { goBack, navigateTo };
}
