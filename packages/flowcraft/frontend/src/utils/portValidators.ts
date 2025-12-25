import { flowcraft } from "../generated/flowcraft";
import { type Edge } from "@xyflow/react";

export interface ConnectionResult {
  canConnect: boolean;
  reason?: string;
}

export interface PortValidator {
  /**
   * 是否允许两个端口连接（类型检查）
   */
  canAccept(
    sourceType: flowcraft.v1.IPortType,
    targetType: flowcraft.v1.IPortType,
  ): boolean;

  /**
   * 该端口允许的最大输入连接数
   */
  getMaxInputs(): number;
}

// --- 1. Standard Validator (Default behavior) ---
export const StandardValidator: PortValidator = {
  canAccept: (src, tgt) => {
    if (src.isGeneric || tgt.isGeneric) return true;
    return src.mainType === tgt.mainType;
  },
  getMaxInputs: () => 1,
};

// --- 2. Collection Validator (List / Set) ---
export const CollectionValidator: PortValidator = {
  canAccept: (src, tgt) => {
    // 允许 相同集合类型 且 内部元素类型匹配
    if (src.mainType === tgt.mainType) {
      return src.itemType === tgt.itemType || !src.itemType || !tgt.itemType;
    }
    // 也允许将 单个元素 接入 集合端口 (自动装箱语义)
    return src.mainType === tgt.itemType;
  },
  getMaxInputs: () => 999, // 允许无限输入
};

// --- 3. Any Validator (Universal) ---
export const AnyValidator: PortValidator = {
  canAccept: () => true, // 接受任何东西
  getMaxInputs: () => 1,
};

/**
 * Validator Registry / Factory
 */
export const getValidator = (
  portType: flowcraft.v1.IPortType | undefined | null,
): PortValidator => {
  const mainType = portType?.mainType || "any";

  if (mainType === "any") return AnyValidator;
  if (mainType === "list" || mainType === "set") return CollectionValidator;

  return StandardValidator;
};

/**
 * Integrated connection check logic
 */
export const validateConnection = (
  source: flowcraft.v1.IPort & { nodeId: string },
  target: flowcraft.v1.IPort & { nodeId: string },
  currentEdges: Edge[],
): ConnectionResult => {
  const validator = getValidator(target.type);

  // 1. Type Check
  if (
    source.type &&
    target.type &&
    !validator.canAccept(source.type, target.type)
  ) {
    return {
      canConnect: false,
      reason: `Type Mismatch: Cannot connect ${source.type.mainType} to ${target.type.mainType}`,
    };
  }

  // 2. Multi-connection Check
  const inputCount = currentEdges.filter(
    (e) => e.target === target.nodeId && e.targetHandle === target.id,
  ).length;
  if (inputCount >= validator.getMaxInputs()) {
    return {
      canConnect: false,
      reason: `Port Full: This input only accepts ${validator.getMaxInputs()} connection(s)`,
    };
  }

  return { canConnect: true };
};
