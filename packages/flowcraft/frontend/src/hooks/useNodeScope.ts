import { useCallback } from "react";
import { useShallow } from "zustand/react/shallow";

import { useFlowStore } from "@/store/flowStore";
import { useUiStore } from "@/store/uiStore";

/**
 * 通用作用域管理 Hook
 * 为“镜头节点”提供穿梭、提取、同步等能力
 */
export const useNodeScope = (nodeId: string, managedScopeId?: string) => {
  const setActiveScope = useUiStore((s) => s.setActiveScope);
  const { allNodes, reparentNode } = useFlowStore(
    useShallow((s) => ({
      allNodes: s.allNodes,
      reparentNode: s.reparentNode,
    })),
  );

  const scopeId = managedScopeId || nodeId;

  // 1. 进入子图 (穿梭)
  const enter = useCallback(() => {
    setActiveScope(scopeId);
  }, [scopeId, setActiveScope]);

  // 2. 提取子节点到当前层级 (脱离环境)
  const extractNode = useCallback(
    (childId: string) => {
      const parent = allNodes.find((n) => n.id === nodeId);
      reparentNode(childId, parent?.parentId || null);
    },
    [nodeId, allNodes, reparentNode],
  );

  // 3. 吸收外部节点到此环境 (归入环境)
  const absorbNode = useCallback(
    (externalId: string) => {
      reparentNode(externalId, scopeId);
    },
    [scopeId, reparentNode],
  );

  return {
    absorbNode,
    enter,
    extractNode,
    isActive: useUiStore((s) => s.activeScopeId === scopeId),
    scopeId,
  };
};
