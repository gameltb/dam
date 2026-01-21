import { PortMainType } from "@/generated/flowcraft/v1/core/base_pb";
import { PortStyle } from "@/generated/flowcraft/v1/core/node_pb";
import { type ClientPort } from "@/types";

/**
 * 核心实践：通过语义化的 PortType 派生 UI 样式，而不是在 Proto 中硬编码颜色。
 */
export const getPortColor = (type?: ClientPort["type"]): string => {
  if (!type) return "var(--port-color-default, #9e9e9e)";

  // 使用 mainType 作为主要 Key
  const typeMap: Partial<Record<PortMainType, string>> = {
    [PortMainType.AUDIO]: "var(--port-color-audio, #3f51b5)",
    [PortMainType.BOOLEAN]: "var(--port-color-boolean, #f44336)",
    [PortMainType.IMAGE]: "var(--port-color-image, #9c27b0)",
    [PortMainType.NUMBER]: "var(--port-color-number, #2196f3)",
    [PortMainType.STRING]: "var(--port-color-string, #4caf50)",
    [PortMainType.SYSTEM]: "var(--port-color-exec, #ffffff)",
    [PortMainType.VIDEO]: "var(--port-color-video, #673ab7)",
  };

  const baseColor = (typeMap as any)[type.mainType as any] ?? "var(--port-color-default, #9e9e9e)";

  // 如果是 generic (泛型) 类型，可以增加一些视觉特征，例如降低透明度
  return type.isGeneric ? `${baseColor}88` : baseColor;
};

/**
 * 根据插槽类型建议最佳形状
 */
export const getPortShape = (type?: ClientPort["type"]): PortStyle => {
  if (!type) return PortStyle.CIRCLE;

  if (type.mainType === PortMainType.SYSTEM) {
    return PortStyle.DASH; // 执行流通常使用特殊形状
  }

  if (type.isGeneric) {
    return PortStyle.SQUARE; // 列表类型使用方块
  }

  return PortStyle.CIRCLE;
};
