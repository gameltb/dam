import { type DescMessage } from "@bufbuild/protobuf";
import React from "react";

import { type AppNode, type DynamicNodeData } from "@/types";

/**
 * Interface for a custom node implementation.
 * Registration is now strictly driven by Protobuf DescMessage.
 */
export interface NodeImplementation {
  // The React component to render
  component: React.ComponentType<{
    data: DynamicNodeData;
    id: string;
    node: AppNode;
    selected?: boolean;
  }>;

  // Optional styling/constraints
  constraints?: {
    minHeight?: number;
    minWidth?: number;
  };

  // Protobuf Schema descriptor for the state of this node
  schema: DescMessage;
}

const registry = new Map<string, NodeImplementation>();

/**
 * Register a specialized node component using its Protobuf Schema.
 * The key is automatically derived from schema.typeName.
 */
export function registerNode(impl: NodeImplementation) {
  const typeName = impl.schema.typeName;
  registry.set(typeName, impl);
}

/**

 * Resolve the component implementation based on node data.

 */

export function resolveNodeComponent(data: DynamicNodeData): NodeImplementation | undefined {
  // Priority 1: Protobuf extension value type (most specific)

  if (data.extension?.value && (data.extension.value as any).$typeName) {
    const typeName = (data.extension.value as any).$typeName;

    const impl = registry.get(typeName);

    if (impl) return impl;
  }

  // Priority 2: Extension case alias (reliable fallback when metadata is lost)

  if (data.extension?.case) {
    const impl = registry.get(data.extension.case);

    if (impl) return impl;
  }

  // Priority 3: templateId (generic fallback)

  if (data.templateId) {
    return registry.get(data.templateId);
  }

  return undefined;
}
