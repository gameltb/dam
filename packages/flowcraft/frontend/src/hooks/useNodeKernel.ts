import { useCallback } from "react";
import { toast } from "react-hot-toast";

import { NodeKernel } from "@/kernel/NodeKernel";
import { useFlowStore } from "@/store/flowStore";

export function useNodeKernel() {
  const spacetimeConn = useFlowStore((s) => s.spacetimeConn);

  const submitTask = useCallback(
    async (
      nodeId: string,
      taskType: string,
      params: any,
      selector?: { preferredWorkerId?: string; requiredCapability?: string },
    ) => {
      if (!spacetimeConn) {
        toast.error("Database not connected");
        return null;
      }

      const kernel = new NodeKernel(spacetimeConn);

      // 1. Guard check
      const isBusy = await kernel.checkNodeBusy(nodeId);
      if (isBusy) {
        toast.error("Node is busy executing another task");
        return null;
      }

      const taskId = crypto.randomUUID();

      // 2. Submit
      try {
        await spacetimeConn.pbreducers.submitTask({
          task: {
            createdAt: BigInt(Date.now()),
            nodeId,
            paramsPayload: new TextEncoder().encode(JSON.stringify(params)),
            selector: selector
              ? {
                  matchTags: {},
                  preferredWorkerId: selector.preferredWorkerId || "",
                  requiredCapability: selector.requiredCapability || "",
                }
              : undefined,
            taskId,
            taskType,
          },
        });
        return taskId;
      } catch (err: any) {
        console.error("Task submission failed", err);
        toast.error(`Failed to submit task: ${err.message}`);
        return null;
      }
    },
    [spacetimeConn],
  );

  return { submitTask };
}
