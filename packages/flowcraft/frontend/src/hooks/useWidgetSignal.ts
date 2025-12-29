import { type WidgetSignal } from "../generated/core/signals_pb";
import { useEffect, useCallback } from "react";
import { useFlowStore, registerWidgetSignalListener } from "../store/flowStore";

export const useWidgetSignal = (nodeId: string, widgetId: string) => {
  const sendWidgetSignal = useFlowStore((s) => s.sendWidgetSignal);

  const sendSignal = useCallback(
    (payload: Record<string, unknown>) => {
      sendWidgetSignal({
        nodeId,
        widgetId,
        payloadJson: JSON.stringify(payload),
      } as unknown as WidgetSignal);
    },
    [nodeId, widgetId, sendWidgetSignal],
  );

  useEffect(() => {
    return registerWidgetSignalListener(nodeId, widgetId, (signal) => {
      console.log("Received widget signal", signal);
    });
  }, [nodeId, widgetId]);

  return { sendSignal };
};
