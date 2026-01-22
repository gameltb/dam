import { type ReducerCtx, t } from "spacetimedb/server";

import type { Viewport as ProtoViewport } from "../generated/flowcraft/v1/core/base_pb";
import type { Edge as ProtoEdge, Node as ProtoNode } from "../generated/flowcraft/v1/core/node_pb";
import type {
  PathUpdateRequest as ProtoPathUpdate,
  ReparentNodeRequest as ProtoReparent,
} from "../generated/flowcraft/v1/core/service_pb";

import { ViewportSchema } from "../generated/flowcraft/v1/core/base_pb";
import { EdgeSchema, NodeSchema } from "../generated/flowcraft/v1/core/node_pb";
import {
  PathUpdateRequestSchema,
  ReparentNodeRequestSchema as ServiceReparentSchema,
} from "../generated/flowcraft/v1/core/service_pb";
import {
  core_Edge as StdbEdge,
  core_Node as StdbNode,
  core_Viewport as StdbViewport,
} from "../generated/generated_schema";
import { pbToStdb } from "../generated/proto-stdb-bridge";
import { type AppSchema } from "../schema";
import { applyPathToObj, unwrapPbValue } from "../utils/path-utils";
import { validateValueByPath } from "../utils/type-validator";

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

  /**
   * 核心：增量路径更新 Reducer
   */
  path_update_pb: {
    args: { req: PathUpdateRequestSchema },
    handler: (ctx: ReducerCtx<AppSchema>, { req }: { req: ProtoPathUpdate }) => {
      const nodeRow = ctx.db.nodes.nodeId.find(req.targetId);
      if (nodeRow) {
        const updated = { ...nodeRow };
        const pathParts = req.path.split(".");

        // 1. 校验路径与类型
        validateValueByPath(NodeSchema, pathParts, req.value);

        // 2. 解包
        const realValue = unwrapPbValue(req.value);

        // 3. 应用补丁 (Zero-Mapping: 路径直接对齐)
        updated.state = applyPathToObj(updated.state, pathParts, realValue, req.type, NodeSchema);

        // 4. 持久化
        ctx.db.nodes.nodeId.update(updated);
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
      const edges = [...ctx.db.edges.iter()];
      for (const edge of edges) {
        const edgeState = edge.state;
        if (edgeState.sourceNodeId === id || edgeState.targetNodeId === id) {
          ctx.db.edges.edgeId.delete(edge.edgeId);
        }
      }
    },
  },

  reparent_node_pb: {
    args: { req: ServiceReparentSchema },
    handler: (ctx: ReducerCtx<AppSchema>, { req }: { req: ProtoReparent }) => {
      const nodeRow = ctx.db.nodes.nodeId.find(req.nodeId);
      if (nodeRow) {
        const updated = { ...nodeRow };
        const state = updated.state;
        if (state.presentation) {
          state.presentation.parentId = req.newParentId;
          if (req.newPosition && state.presentation.position) {
            state.presentation.position.x = req.newPosition.x;
            state.presentation.position.y = req.newPosition.y;
          }
        }
        ctx.db.nodes.nodeId.update(updated);
      }
    },
  },

  update_viewport: {
    args: { id: t.string(), viewport: ViewportSchema },
    handler: (ctx: ReducerCtx<AppSchema>, { id, viewport }: { id: string; viewport: ProtoViewport }) => {
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
};
