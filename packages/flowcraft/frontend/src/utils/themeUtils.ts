import {
  type PortType,
  PortStyle,
} from "../generated/flowcraft/v1/core/node_pb";
import { PROTO_TO_PORT_TYPE } from "./protoAdapter";

/**
 * 核心实践：通过语义化的 PortType 派生 UI 样式，而不是在 Proto 中硬编码颜色。
 */
export const getPortColor = (type?: PortType): string => {
  if (!type) return "var(--port-color-default, #9e9e9e)";

  // 使用 mainType 作为主要 Key
  const typeMap: Record<string, string> = {
    string: "var(--port-color-string, #4caf50)",
    number: "var(--port-color-number, #2196f3)",
    boolean: "var(--port-color-boolean, #f44336)",
    image: "var(--port-color-image, #9c27b0)",
    video: "var(--port-color-video, #673ab7)",
    audio: "var(--port-color-audio, #3f51b5)",
    model: "var(--port-color-model, #ff9800)",
    tensor: "var(--port-color-tensor, #ffeb3b)",
    exec: "var(--port-color-exec, #ffffff)",
  };

  const mainTypeStr =
    PROTO_TO_PORT_TYPE[type.mainType as unknown as number] ?? "any";
  const baseColor =
    typeMap[mainTypeStr.toLowerCase()] ?? "var(--port-color-default, #9e9e9e)";

  // 如果是 generic (泛型) 类型，可以增加一些视觉特征，例如降低透明度
  return type.isGeneric ? `${baseColor}88` : baseColor;
};

/**
 * 根据插槽类型建议最佳形状
 */
export const getPortShape = (type?: PortType): PortStyle => {
  if (!type) return PortStyle.CIRCLE;

  const mainTypeStr =
    PROTO_TO_PORT_TYPE[type.mainType as unknown as number] ?? "any";
  if (mainTypeStr.toLowerCase() === "exec") {
    return PortStyle.DASH; // 执行流通常使用特殊形状
  }

  if (type.itemType) {
    return PortStyle.SQUARE; // 列表类型使用方块
  }

  return PortStyle.CIRCLE;
};
