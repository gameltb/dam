import { type ReducerCtx, t } from "spacetimedb/server";

import { ViewportSchema } from "../generated/flowcraft/v1/core/base_pb";
import { Viewport as ProtoViewport } from "../generated/flowcraft/v1/core/base_pb";
import { Edge as ProtoEdge, Node as ProtoNode } from "../generated/flowcraft/v1/core/node_pb";
import { EdgeSchema, NodeSchema } from "../generated/flowcraft/v1/core/node_pb";
import { Edge as StdbEdge, Node as StdbNode, Viewport as StdbViewport } from "../generated/generated_schema";
import { pbToStdb } from "../generated/proto-stdb-bridge";
import { type AppSchema } from "../schema";

export const nodeReducers = {
  add_edge_pb: {
    args: { edge: EdgeSchema },
    handler: (ctx: ReducerCtx<AppSchema>, { edge }: { edge: ProtoEdge }) => {
      ctx.db.edges.insert({
        edgeId: edge.edgeId,
        state: pbToStdb(EdgeSchema, StdbEdge, edge) as StdbEdge,
      });
    },
  },

  create_node_pb: {
    args: { node: NodeSchema },
    handler: (ctx: ReducerCtx<AppSchema>, { node }: { node: ProtoNode }) => {
      ctx.db.nodes.insert({
        nodeId: node.nodeId,
        state: pbToStdb(NodeSchema, StdbNode, node) as StdbNode,
      });
    },
  },

  move_node: {
    args: { id: t.string(), x: t.f64(), y: t.f64() },
    handler: (ctx: ReducerCtx<AppSchema>, { id, x, y }: { id: string; x: number; y: number }) => {
      const nodeRow = ctx.db.nodes.nodeId.find(id);
      if (nodeRow?.state.presentation?.position) {
        const updated = { ...nodeRow };
        const presentation = updated.state.presentation;
        if (presentation?.position) {
          presentation.position.x = x;
          presentation.position.y = y;
          ctx.db.nodes.nodeId.update(updated);
        }
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
      ctx.db.nodes.nodeId.delete(id);
      // Clean up connected edges
      const edges = [...ctx.db.edges.iter()];
      for (const edge of edges) {
        const edgeState = edge.state;
        if (edgeState.sourceNodeId === id || edgeState.targetNodeId === id) {
          ctx.db.edges.edgeId.delete(edge.edgeId);
        }
      }
    },
  },

  update_node_pb: {
    args: { node: NodeSchema },
    handler: (ctx: ReducerCtx<AppSchema>, { node }: { node: ProtoNode }) => {
      const existing = ctx.db.nodes.nodeId.find(node.nodeId);
      if (existing) {
        ctx.db.nodes.nodeId.update({
          nodeId: node.nodeId,
          state: pbToStdb(NodeSchema, StdbNode, node) as StdbNode,
        });
      }
    },
  },

  update_viewport: {
    args: { id: t.string(), viewport: ViewportSchema },
    handler: (
      ctx: ReducerCtx<AppSchema>,
      { id, viewport }: { id: string; viewport: ProtoViewport },
    ) => {
      const existing = ctx.db.viewportState.id.find(id);
      if (existing) {
        ctx.db.viewportState.id.update({
          id,
          state: pbToStdb(ViewportSchema, StdbViewport, viewport) as StdbViewport,
        });
      } else {
        ctx.db.viewportState.insert({
          id,
          state: pbToStdb(ViewportSchema, StdbViewport, viewport) as StdbViewport,
        });
      }
    },
  },

  update_widget_value: {
    args: { nodeId: t.string(), value: t.string(), widgetId: t.string() },
    handler: (
      ctx: ReducerCtx<AppSchema>,
      { nodeId, value, widgetId }: { nodeId: string; value: string; widgetId: string },
    ) => {
      const id = `${nodeId}_${widgetId}`;
      const existing = ctx.db.widgetValues.id.find(id);
      if (existing) {
        ctx.db.widgetValues.id.update({ id, nodeId, value, widgetId });
      } else {
        ctx.db.widgetValues.insert({ id, nodeId, value, widgetId });
      }
    },
  },
};
