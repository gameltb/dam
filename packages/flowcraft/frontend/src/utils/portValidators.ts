import { type Port, type PortType } from "../generated/flowcraft/v1/node_pb";
import { PortMainType } from "../generated/flowcraft/v1/base_pb";
import { PROTO_TO_PORT_TYPE } from "./protoAdapter";
import type { Edge } from "@xyflow/react";

export interface ConnectionResult {
  canConnect: boolean;
  reason?: string;
}

export interface PortValidator {
  /**
   * 是否允许两个端口连接（类型检查）
   */
  canAccept(sourceType: PortType, targetType: PortType): boolean;

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
    return PROTO_TO_PORT_TYPE[src.mainType] === tgt.itemType;
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
  portType: PortType | undefined | null,
): PortValidator => {
  const mainType = portType?.mainType ?? PortMainType.ANY;

  if (mainType === PortMainType.ANY) return AnyValidator;
  if (mainType === PortMainType.LIST || mainType === PortMainType.SET)
    return CollectionValidator;

  return StandardValidator;
};

/**
 * Integrated connection check logic
 */
export const validateConnection = (
  source: Port & { nodeId: string },
  target: Port & { nodeId: string },
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
      reason: `Type Mismatch: Cannot connect ${PROTO_TO_PORT_TYPE[source.type.mainType] ?? "any"} to ${PROTO_TO_PORT_TYPE[target.type.mainType] ?? "any"}`,
    };
  }

  // 2. Multi-connection Check
  const maxInputs = validator.getMaxInputs();
  const inputCount = currentEdges.filter(
    (e) => e.target === target.nodeId && e.targetHandle === target.id,
  ).length;

  if (inputCount >= maxInputs) {
    // If it's a single input port, we allow connection to "replace" the existing one
    if (maxInputs === 1) {
      return { canConnect: true };
    }

    return {
      canConnect: false,
      reason: `Port Full: This input only accepts ${String(maxInputs)} connection(s)`,
    };
  }

  return { canConnect: true };
};
