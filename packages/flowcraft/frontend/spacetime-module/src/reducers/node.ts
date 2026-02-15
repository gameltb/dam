import { toBinary } from "@bufbuild/protobuf";
import { type ReducerCtx, t } from "spacetimedb/server";

import { ViewportSchema, type Viewport } from "../generated/flowcraft/v1/core/base_pb";
import {
  type Edge,
  EdgeSchema,
  type Node,
  type NodeData,
  NodeDataSchema,
  NodeSchema,
} from "../generated/flowcraft/v1/core/node_pb";
import { type AddSubGraphRequest, AddSubGraphRequestSchema } from "../generated/flowcraft/v1/core/service_pb";
import { type AppSchema } from "../schema";

export const nodeReducers = {
  add_edge_pb: {
    args: { edge: EdgeSchema },
    handler: (ctx: ReducerCtx<AppSchema>, { edge, edgeBinary }: { edge: Edge; edgeBinary: Uint8Array }) => {
      ctx.db.edges.insert({
        edgeId: edge.edgeId,
        graphId: edge.graphId,
        sourceNodeId: edge.sourceNodeId,
        state: edgeBinary,
        targetNodeId: edge.targetNodeId,
      });
    },
  },

  /**
   * Batch import subgraph: handles combinations of multiple nodes and edges.
   */
  add_sub_graph_pb: {
    args: { req: AddSubGraphRequestSchema },
    handler: (ctx: ReducerCtx<AppSchema>, { req }: { req: AddSubGraphRequest }) => {
      // 1. Batch create nodes
      req.nodes.forEach((node) => {
        const nodeId = node.nodeId;
        const graphId = node.graphId || "default";

        // Identity
        ctx.db.nodes.insert({
          graphId,
          nodeId,
          nodeKind: node.nodeKind as unknown as number,
          templateId: node.templateId,
        });

        // Transform
        ctx.db.nodeTransforms.insert({
          height: node.presentation?.height ?? 0,
          nodeId,
          width: node.presentation?.width ?? 0,
          x: node.presentation?.position?.x ?? 0,
          y: node.presentation?.position?.y ?? 0,
        });

        // Metadata: Correctly mapping scopeId and parentId
        ctx.db.nodeMetadata.insert({
          displayName: node.state?.displayName ?? "",
          graphId,
          nodeId,
          parentId: node.presentation?.parentId ?? "",
          scopeId: node.presentation?.scopeId ?? "root",
        });

        // State Blob
        if (node.state) {
          ctx.db.nodeData.insert({
            nodeId,
            state: toBinary(NodeDataSchema, node.state),
          });
        }
      });

      // 2. Batch create edges
      req.edges.forEach((edge) => {
        ctx.db.edges.insert({
          edgeId: edge.edgeId,
          graphId: edge.graphId || "default",
          sourceNodeId: edge.sourceNodeId,
          state: toBinary(EdgeSchema, edge),
          targetNodeId: edge.targetNodeId,
        });
      });
    },
  },

  /**
   * Create node: writes to multiple component tables simultaneously.
   */
  create_node_pb: {
    args: { node: NodeSchema },
    handler: (ctx: ReducerCtx<AppSchema>, { node }: { node: Node }) => {
      const nodeId = node.nodeId;
      const graphId = node.graphId || "default";

      // 1. Identity
      ctx.db.nodes.insert({
        graphId,
        nodeId,
        nodeKind: node.nodeKind as unknown as number,
        templateId: node.templateId,
      });

      // 2. Transform
      ctx.db.nodeTransforms.insert({
        height: node.presentation?.height ?? 0,
        nodeId,
        width: node.presentation?.width ?? 0,
        x: node.presentation?.position?.x ?? 0,
        y: node.presentation?.position?.y ?? 0,
      });

      // 3. Metadata: Supporting both scope and parent hierarchy
      ctx.db.nodeMetadata.insert({
        displayName: node.state?.displayName ?? "",
        graphId,
        nodeId,
        parentId: node.presentation?.parentId ?? "",
        scopeId: node.presentation?.scopeId ?? "root",
      });

      // 4. State Blob (NodeData)
      if (node.state) {
        ctx.db.nodeData.insert({
          nodeId,
          state: toBinary(NodeDataSchema, node.state),
        });
      }
    },
  },

  remove_edge: {
    args: { id: t.string() },
    handler: (ctx: ReducerCtx<AppSchema>, { id }: { id: string }) => {
      ctx.db.edges.edgeId.delete(id);
    },
  },

  remove_node: {
    args: { id: t.string() },
    handler: (ctx: ReducerCtx<AppSchema>, { id }: { id: string }) => {
      // Cascade delete all components
      ctx.db.nodes.nodeId.delete(id);
      ctx.db.nodeTransforms.nodeId.delete(id);
      ctx.db.nodeMetadata.nodeId.delete(id);
      ctx.db.nodeData.nodeId.delete(id);

      // Delete related edges
      for (const edge of ctx.db.edges.sourceNodeId.filter(id)) {
        ctx.db.edges.edgeId.delete(edge.edgeId);
      }
      for (const edge of ctx.db.edges.targetNodeId.filter(id)) {
        ctx.db.edges.edgeId.delete(edge.edgeId);
      }
    },
  },

  /**
   * Only update core PB data.
   */
  set_node_data_pb: {
    args: { nodeId: t.string(), state: NodeDataSchema },
    handler: (
      ctx: ReducerCtx<AppSchema>,
      { nodeId, state, stateBinary }: { nodeId: string; state: NodeData; stateBinary: Uint8Array },
    ) => {
      const existing = ctx.db.nodeData.nodeId.find(nodeId);
      if (existing) {
        ctx.db.nodeData.nodeId.update({ nodeId, state: stateBinary });
      } else {
        ctx.db.nodeData.insert({ nodeId, state: stateBinary });
      }

      // Sync update displayName to Metadata table for consistency
      const metadata = ctx.db.nodeMetadata.nodeId.find(nodeId);
      if (metadata && state.displayName !== undefined) {
        ctx.db.nodeMetadata.nodeId.update({ ...metadata, displayName: state.displayName });
      }
    },
  },

  /**
   * Update physical parent node (used for hierarchy changes within a Group).
   */
  set_node_parent: {
    args: { nodeId: t.string(), parentId: t.string() },
    handler: (ctx: ReducerCtx<AppSchema>, { nodeId, parentId }: { nodeId: string; parentId: string }) => {
      const metadata = ctx.db.nodeMetadata.nodeId.find(nodeId);
      if (metadata) {
        ctx.db.nodeMetadata.nodeId.update({ ...metadata, parentId });
      }
    },
  },

  /**
   * High frequency: only update coordinates.
   */
  set_node_position: {
    args: { nodeId: t.string(), x: t.f32(), y: t.f32() },
    handler: (ctx: ReducerCtx<AppSchema>, { nodeId, x, y }: { nodeId: string; x: number; y: number }) => {
      const transform = ctx.db.nodeTransforms.nodeId.find(nodeId);
      if (transform) {
        ctx.db.nodeTransforms.nodeId.update({ ...transform, x, y });
      }
    },
  },

  /**
   * Update logical Scope (used for moving across levels).
   */
  set_node_scope: {
    args: { nodeId: t.string(), scopeId: t.string() },
    handler: (ctx: ReducerCtx<AppSchema>, { nodeId, scopeId }: { nodeId: string; scopeId: string }) => {
      const metadata = ctx.db.nodeMetadata.nodeId.find(nodeId);
      if (metadata) {
        ctx.db.nodeMetadata.nodeId.update({ ...metadata, scopeId });
      }
    },
  },

  /**
   * High frequency: only update dimensions.
   */
  set_node_size: {
    args: { height: t.f32(), nodeId: t.string(), width: t.f32() },
    handler: (
      ctx: ReducerCtx<AppSchema>,
      { height, nodeId, width }: { height: number; nodeId: string; width: number },
    ) => {
      const transform = ctx.db.nodeTransforms.nodeId.find(nodeId);
      if (transform) {
        ctx.db.nodeTransforms.nodeId.update({ ...transform, height, width });
      }
    },
  },

  update_viewport: {
    args: { id: t.string(), viewport: ViewportSchema },
    handler: (ctx: ReducerCtx<AppSchema>, { id, viewport, viewportBinary }: { id: string; viewport: Viewport; viewportBinary: Uint8Array }) => {
      const existing = ctx.db.viewportState.id.find(id);
      const graphId = viewport.graphId || "default";
      if (existing) {
        ctx.db.viewportState.id.update({ graphId, id, state: viewportBinary });
      } else {
        ctx.db.viewportState.insert({ graphId, id, state: viewportBinary });
      }
    },
  },

  updateWidgetValue: {
    args: { id: t.string(), nodeId: t.string(), value: t.string(), widgetId: t.string() },
    handler: (ctx: ReducerCtx<AppSchema>, params: { id: string; nodeId: string; value: string; widgetId: string }) => {
      const existing = ctx.db.widgetValues.id.find(params.id);
      if (existing) {
        ctx.db.widgetValues.id.update(params);
      } else {
        ctx.db.widgetValues.insert(params);
      }
    },
  },
};
