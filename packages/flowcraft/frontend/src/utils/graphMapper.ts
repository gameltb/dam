import { create } from "@bufbuild/protobuf";
import { type Infer } from "spacetimedb";

import { PositionSchema } from "@/generated/flowcraft/v1/core/base_pb";
import {
  EdgesRow,
  NodeDataRow,
  NodeMetadataRow,
  NodesRow,
  NodeTransformsRow,
  ViewportStateRow,
} from "@/generated/spacetime";
import { type AppEdge, type AppNode, AppNodeType, type DynamicNodeData, NodeStatus, Scope } from "@/types";

import { convertStdbToPb } from "./pb-client";

/**
 * GraphMapper (V8.0)
 *
 * Supports separate scopeId (logical) and parentId (physical).
 */
export const GraphMapper = {
  applyData(node: AppNode, dataRow: Infer<typeof NodeDataRow>): AppNode {
    const pbData = convertStdbToPb("nodeData", dataRow as unknown as Record<string, unknown>);
    return {
      ...node,
      _lastSync: Date.now(),
      data: pbData as DynamicNodeData,
      scopeId: node.scopeId ?? Scope.ROOT,
    };
  },

  applyMetadata(node: AppNode, metadataRow: Infer<typeof NodeMetadataRow>): AppNode {
    return {
      ...node,
      _lastSync: Date.now(),
      data: {
        ...node.data,
        displayName: metadataRow.displayName,
      },
      parentId: metadataRow.parentId || undefined,
      presentation: node.presentation
        ? {
            ...node.presentation,
            parentId: metadataRow.parentId ?? "",
            scopeId: metadataRow.scopeId || Scope.ROOT,
          }
        : undefined,
      scopeId: metadataRow.scopeId || Scope.ROOT,
    };
  },

  applyTransform(node: AppNode, transformRow: Infer<typeof NodeTransformsRow>): AppNode {
    const x = typeof transformRow.x === "number" && !isNaN(transformRow.x) ? transformRow.x : (node.position?.x ?? 0);
    const y = typeof transformRow.y === "number" && !isNaN(transformRow.y) ? transformRow.y : (node.position?.y ?? 0);

    return {
      ...node,
      _lastSync: Date.now(),
      height: transformRow.height || node.height || 200,
      position: { x, y },
      presentation: node.presentation
        ? {
            ...node.presentation,
            height: transformRow.height || node.presentation.height,
            position: create(PositionSchema, { x, y }),
            width: transformRow.width || node.presentation.width,
          }
        : ({
            // Ensure presentation exists if transform exists
            height: transformRow.height,
            isInitialized: true,
            position: create(PositionSchema, { x, y }),
            width: transformRow.width,
          } as any),
      width: transformRow.width || node.width || 300,
    };
  },

  createSkeleton(nodeRow: Infer<typeof NodesRow>): AppNode {
    return {
      _lastSync: Date.now(),
      data: {
        activeMode: 0,
        availableModes: [],
        displayName: "Loading…",
        extension: { case: undefined, value: undefined },
        inputPorts: [],
        metadata: {},
        outputPorts: [],
        schemaVersion: 1,
        status: NodeStatus.IDLE,
        taskId: "",
        widgets: [],
      } as unknown as DynamicNodeData, // Initial loading state
      graphId: nodeRow.graphId,
      height: 200,
      id: nodeRow.nodeId,
      position: { x: 0, y: 0 },
      scopeId: Scope.ROOT, // Defaults to root until metadata arrives
      type: nodeRow.templateId === "chatMessage" ? AppNodeType.CHAT_MESSAGE : AppNodeType.DYNAMIC,
      width: 300,
    };
  },

  toEdge(edgeRow: Infer<typeof EdgesRow>): AppEdge {
    const pbEdge = convertStdbToPb("edges", edgeRow as unknown as Record<string, unknown>);
    return {
      data: pbEdge.metadata,
      graphId: edgeRow.graphId,
      id: pbEdge.edgeId,
      source: pbEdge.sourceNodeId,
      sourceHandle: pbEdge.sourceHandle || undefined,
      target: pbEdge.targetNodeId,
      targetHandle: pbEdge.targetHandle || undefined,
    };
  },

  toViewport(entry: Infer<typeof ViewportStateRow>) {
    const remote = convertStdbToPb("viewportState", entry as unknown as Record<string, unknown>);
    return remote && typeof remote.x === "number" && !isNaN(remote.x)
      ? { x: remote.x, y: remote.y, zoom: remote.zoom }
      : null;
  },
};
