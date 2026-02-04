import { create } from "@bufbuild/protobuf";
import { type Edge as RFEdge } from "@xyflow/react";
import { type Infer } from "spacetimedb";

import { PositionSchema } from "@/generated/flowcraft/v1/core/base_pb";
import {
  EdgesRow,
  NodeDataRow,
  NodeMetadataRow,
  NodeTransformsRow,
  NodesRow,
  ViewportStateRow,
} from "@/generated/spacetime";
import { type AppNode, AppNodeType, Scope } from "@/types";

import { convertStdbToPb } from "./pb-client";

/**
 * GraphMapper (V8.0)
 *
 * Supports separate scopeId (logical) and parentId (physical).
 */
export const GraphMapper = {
  applyData(node: AppNode, dataRow: Infer<typeof NodeDataRow>): AppNode {
    const pbData = convertStdbToPb("nodeData", dataRow);
    return {
      ...node,
      _lastSync: Date.now(),
      data: pbData,
      scopeId: node.scopeId || Scope.ROOT,
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
            parentId: metadataRow.parentId || "",
            scopeId: metadataRow.scopeId || Scope.ROOT,
          }
        : undefined,
      scopeId: metadataRow.scopeId || Scope.ROOT,
    };
  },

  applyTransform(node: AppNode, transformRow: Infer<typeof NodeTransformsRow>): AppNode {
    return {
      ...node,
      _lastSync: Date.now(),
      height: transformRow.height,
      position: { x: transformRow.x, y: transformRow.y },
      presentation: node.presentation
        ? {
            ...node.presentation,
            height: transformRow.height,
            position: create(PositionSchema, { x: transformRow.x, y: transformRow.y }),
            width: transformRow.width,
          }
        : undefined,
      width: transformRow.width,
    };
  },

  createSkeleton(nodeRow: Infer<typeof NodesRow>): AppNode {
    return {
      _lastSync: Date.now(),
      data: { availableModes: [], displayName: "Loading…" } as any, // Initial loading state
      height: 200,
      id: nodeRow.nodeId,
      position: { x: 0, y: 0 },
      scopeId: Scope.ROOT, // Defaults to root until metadata arrives
      type: nodeRow.templateId === "chatMessage" ? AppNodeType.CHAT_MESSAGE : AppNodeType.DYNAMIC,
      width: 300,
    };
  },

  toEdge(edgeRow: Infer<typeof EdgesRow>): RFEdge {
    const pbEdge = convertStdbToPb("edges", edgeRow);
    return {
      data: pbEdge.metadata,
      id: pbEdge.edgeId,
      source: pbEdge.sourceNodeId,
      sourceHandle: pbEdge.sourceHandle || undefined,
      target: pbEdge.targetNodeId,
      targetHandle: pbEdge.targetHandle || undefined,
    };
  },

  toViewport(entry: Infer<typeof ViewportStateRow>) {
    const remote = convertStdbToPb("viewportState", entry as Record<string, unknown>);
    return remote && !isNaN(remote.x) ? { x: remote.x, y: remote.y, zoom: remote.zoom } : null;
  },
};
