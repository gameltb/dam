import { type WidgetSignal } from "../generated/flowcraft/v1/signals_pb";
import { useEffect, useCallback } from "react";
import { useFlowStore } from "../store/flowStore";
import { registerWidgetSignalListener } from "../store/signalHandlers";

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
    return registerWidgetSignalListener(
      nodeId,
      widgetId,
      (signal: WidgetSignal) => {
        console.log("Received widget signal", signal);
      },
    );
  }, [nodeId, widgetId]);

  return { sendSignal };
};
