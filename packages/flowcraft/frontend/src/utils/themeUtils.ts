import { PortMainType } from "@/generated/flowcraft/v1/core/base_pb";
import { PortStyle } from "@/generated/flowcraft/v1/core/node_pb";
import { type ClientPort } from "@/types";

/**
 * Core Practice: Derive UI styles from semantic PortType instead of hardcoding colors in Proto.
 */
export const getPortColor = (type?: ClientPort["type"]): string => {
  if (!type) return "var(--port-color-default, #9e9e9e)";

  // Use mainType as the primary key
  const typeMap: Partial<Record<PortMainType, string>> = {
    [PortMainType.AUDIO]: "var(--port-color-audio, #3f51b5)",
    [PortMainType.BOOLEAN]: "var(--port-color-boolean, #f44336)",
    [PortMainType.IMAGE]: "var(--port-color-image, #9c27b0)",
    [PortMainType.NUMBER]: "var(--port-color-number, #2196f3)",
    [PortMainType.STRING]: "var(--port-color-string, #4caf50)",
    [PortMainType.SYSTEM]: "var(--port-color-exec, #ffffff)",
    [PortMainType.VIDEO]: "var(--port-color-video, #673ab7)",
  };

  const baseColor = typeMap[type.mainType] ?? "var(--port-color-default, #9e9e9e)";

  // If it's a generic type, we can add some visual characteristics, such as reduced opacity
  return type.isGeneric ? `${baseColor}88` : baseColor;
};

/**
 * Suggest the best shape based on the port type
 */
export const getPortShape = (type?: ClientPort["type"]): PortStyle => {
  if (!type) return PortStyle.CIRCLE;

  if (type.mainType === PortMainType.SYSTEM) {
    return PortStyle.DASH; // Execution flows usually use special shapes
  }

  if (type.isGeneric) {
    return PortStyle.SQUARE; // List types use squares
  }

  return PortStyle.CIRCLE;
};

/**
 * Standard styles for NodeResizer handles
 */
export const RESIZER_COLOR = "var(--primary-color)";
export const RESIZER_HANDLE_STYLE: React.CSSProperties = {
  backgroundColor: "var(--primary-color)",
  border: "2px solid white",
  borderRadius: "50%",
  height: 10,
  width: 10,
};
