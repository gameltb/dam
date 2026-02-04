import { create } from "@bufbuild/protobuf";
import { useCallback, useMemo } from "react";
import { useTable } from "spacetimedb/react";

import { ResetNodeRequestSchema } from "@/generated/flowcraft/v1/core/service_pb";
import { tables } from "@/generated/spacetime";
import { useFlowStore } from "@/store/flowStore";

export interface NodeController {
  error: null | string;
  isOwner: boolean;
  message: string;
  progress: number;
  reset: (clearData?: boolean) => void;
  status: "busy" | "error" | "idle";
}

/**
 * Standard Scaffold Hook for node lifecycle management.
 * Every node implementation should use this to handle its runtime state.
 */
export function useNodeController(nodeId: string): NodeController {
  const [runtimeStates] = useTable(tables.nodeRuntimeStates);
  const spacetimeConn = useFlowStore((s) => s.spacetimeConn);

  const state = useMemo(() => {
    const entry = runtimeStates.find((s) => s.nodeId === nodeId);
    if (!entry) {
      return {
        activeUserId: null,
        error: null,
        message: "",
        progress: 0,
        status: "idle" as const,
      };
    }
    return {
      activeUserId: entry.activeUserId ?? null,
      error: entry.error ?? null,
      message: entry.message,
      progress: entry.progress,
      status: entry.status as "busy" | "error" | "idle",
    };
  }, [runtimeStates, nodeId]);

  const reset = useCallback(
    (clearData = false) => {
      if (spacetimeConn) {
        // Direct call to specialized reset reducer
        // Since ResetNodeRequest was added to FlowMessage, we might need a dedicated reducer in STDB
        // or send it as a message. For now, assuming a reducer resetNode exists or using pbreducers.
        if (spacetimeConn.pbreducers.resetNode) {
          spacetimeConn.pbreducers.resetNode({
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
