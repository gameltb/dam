import { create } from "@bufbuild/protobuf";
import { useCallback, useMemo } from "react";
import { useTable } from "spacetimedb/react";

import { ResetNodeRequestSchema } from "@/generated/flowcraft/v1/core/service_pb";
import { tables } from "@/generated/spacetime";
import { useFlowStore } from "@/store/flowStore";
import { NodeStatus } from "@/types";

export interface NodeController {
  error: null | string;
  isOwner: boolean;
  message: string;
  progress: number;
  reset: (clearData?: boolean) => void;
  status: NodeStatus;
}

/**
 * Standard Scaffold Hook for node lifecycle management.
 * Every node implementation should use this to handle its runtime state.
 */
export function useNodeController(nodeId: string): NodeController {
  const [runtimeStates] = useTable(tables.nodeRuntimeStates);
  const spacetimeConn = useFlowStore((s) => s.spacetimeConn);
  const node = useFlowStore((s) => s.nodesById[nodeId]);

  const state = useMemo(() => {
    const entry = runtimeStates.find((s) => s.nodeId === nodeId);
    const nodeStatus = node?.data?.status ?? NodeStatus.IDLE;

    if (!entry) {
      return {
        activeUserId: null,
        error: null,
        message: "",
        progress: 0,
        status: nodeStatus,
      };
    }

    // Map transient status string to NodeStatus if necessary, or prioritize NodeData.status
    // In our new ECP model, NodeData.status is the source of truth for the state machine.
    return {
      activeUserId: entry.activeUserId ?? null,
      error: entry.error ?? null,
      message: entry.message,
      progress: entry.progress,
      status: nodeStatus,
    };
  }, [runtimeStates, nodeId, node?.data?.status]);

  const reset = useCallback(
    (clearData = false) => {
      if (spacetimeConn) {
        const pbreducers = spacetimeConn.pbreducers as any;
        if (pbreducers.resetNode) {
          pbreducers.resetNode({
            req: create(ResetNodeRequestSchema, {
              clearData,
              nodeId,
            }),
          });
        }
      }
    },
    [spacetimeConn, nodeId],
  );

  return {
    ...state,
    isOwner: true, // Placeholder for ownership logic
    reset,
  };
}
