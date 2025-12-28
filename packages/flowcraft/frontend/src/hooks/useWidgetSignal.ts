import { useEffect, useCallback } from "react";
import { useFlowStore, registerWidgetSignalListener } from "../store/flowStore";
import { flowcraft_proto } from "../generated/flowcraft_proto";

/**
 * Hook for widgets to communicate with the backend using type-safe Proto signals.
 *
 * @param nodeId The ID of the node containing the widget
 * @param widgetId The ID of the widget
 * @param onSignal Callback when a signal is received from the backend
 */
export function useWidgetSignal(
  nodeId: string,
  widgetId: string,
  onSignal?: (signal: flowcraft_proto.v1.IWidgetSignal) => void,
) {
  const sendWidgetSignal = useFlowStore((s) => s.sendWidgetSignal);

  useEffect(() => {
    if (!onSignal) return;
    return registerWidgetSignalListener(nodeId, widgetId, onSignal);
  }, [nodeId, widgetId, onSignal]);

  const sendSignal = useCallback(
    (payload: flowcraft_proto.v1.IWidgetSignal["payload"]) => {
      const signal: flowcraft_proto.v1.IWidgetSignal = {
        node_id: nodeId,
        widget_id: widgetId,
      };

      if (payload) {
        // Correctly map the oneof payload based on the input
        Object.assign(signal, payload);
      }

      sendWidgetSignal(signal);
    },
    [nodeId, widgetId, sendWidgetSignal],
  );

  return { sendSignal };
}
