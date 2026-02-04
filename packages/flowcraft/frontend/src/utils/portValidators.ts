import type { Edge } from "@xyflow/react";

import { PortMainType } from "@/generated/flowcraft/v1/core/base_pb";
import { type Port, type PortType } from "@/generated/flowcraft/v1/core/node_pb";

import { PORT_MAIN_TYPE_FROM_PROTO } from "./nodeUtils";

export interface ConnectionResult {
  canConnect: boolean;
  reason?: string;
}

export interface PortValidator {
  /**
   * Whether to allow two ports to connect (type check).
   */
  canAccept(sourceType: PortType, targetType: PortType): boolean;

  /**
   * The maximum number of input connections allowed for this port.
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
    // Allow same collection type and matching internal element types
    if (src.mainType === tgt.mainType) {
      return src.itemType === tgt.itemType || !src.itemType || !tgt.itemType;
    }
    // Also allow connecting a single element to a collection port (autoboxing semantics)
    return PORT_MAIN_TYPE_FROM_PROTO[src.mainType] === tgt.itemType;
  },
  getMaxInputs: () => 999, // Allow infinite inputs
};

// --- 3. Any Validator (Universal) ---
export const AnyValidator: PortValidator = {
  canAccept: () => true, // Accept anything
  getMaxInputs: () => 1,
};

/**
 * Validator Registry / Factory
 */
export const getValidator = (portType: null | PortType | undefined): PortValidator => {
  const mainType = portType?.mainType ?? PortMainType.ANY;

  if (mainType === PortMainType.ANY) return AnyValidator;
  if (mainType === PortMainType.LIST || mainType === PortMainType.SET) return CollectionValidator;

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
  if (source.type && target.type && !validator.canAccept(source.type, target.type)) {
    return {
      canConnect: false,
      reason: `Type Mismatch: Cannot connect ${PORT_MAIN_TYPE_FROM_PROTO[source.type.mainType] ?? "any"} to ${PORT_MAIN_TYPE_FROM_PROTO[target.type.mainType] ?? "any"}`,
    };
  }

  // 2. Multi-connection Check
  const maxInputs = validator.getMaxInputs();
  const inputCount = currentEdges.filter((e) => e.target === target.nodeId && e.targetHandle === target.id).length;

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
