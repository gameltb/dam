import { type WidgetSignal } from "@/generated/flowcraft/v1/core/signals_pb";

const widgetSignalListeners = new Map<string, (signal: WidgetSignal) => void>();

export const registerWidgetSignalListener = (
  nodeId: string,
  widgetId: string,
  callback: (signal: WidgetSignal) => void,
) => {
  const key = `${nodeId}-${widgetId}`;
  widgetSignalListeners.set(key, callback);
  return () => {
    widgetSignalListeners.delete(key);
  };
};

export const getWidgetSignalListener = (nodeId: string, widgetId: string) => {
  const key = `${nodeId}-${widgetId}`;
  return widgetSignalListeners.get(key);
};
