import { t } from "spacetimedb/server";
import { logOperation } from "../utils/audit";

export const nodeReducers = {
  create_node: {
    args: {
      dataJson: t.string(),
      height: t.f32(),
      id: t.string(),
      isSelected: t.bool(),
      kind: t.u32(),
      parentId: t.string(),
      posX: t.f32(),
      posY: t.f32(),
      templateId: t.string(),
      width: t.f32(),
    },
    handler: (ctx: any, args: any) => {
      ctx.db.nodes.insert(args);
      logOperation(ctx, "create_node", { id: args.id, templateId: args.templateId });
    },
  },

  update_node_data: {
    args: { dataJson: t.string(), id: t.string() },
    handler: (ctx: any, { dataJson, id }: any) => {
      const node = ctx.db.nodes.id.find(id);
      if (node) {
        ctx.db.nodes.id.update({ ...node, dataJson });
        logOperation(ctx, "update_node_data", { dataJson, id });
      }
    },
  },

  move_node: {
    args: { id: t.string(), x: t.f32(), y: t.f32() },
    handler: (ctx: any, { id, x, y }: any) => {
      const node = ctx.db.nodes.id.find(id);
      if (node) {
        ctx.db.nodes.id.update({ ...node, posX: x, posY: y });
        logOperation(ctx, "move_node", { id, x, y });
      }
    },
  },

  remove_node: {
    args: { id: t.string() },
    handler: (ctx: any, { id }: any) => {
      ctx.db.nodes.id.delete(id);
      for (const edge of ctx.db.edges) {
        if (edge.sourceId === id || edge.targetId === id) {
          ctx.db.edges.id.delete(edge.id);
        }
      }
      logOperation(ctx, "remove_node", { id });
    },
  },

  add_edge: {
    args: {
      id: t.string(),
      sourceHandle: t.string(),
      sourceId: t.string(),
      targetHandle: t.string(),
      targetId: t.string(),
    },
    handler: (ctx: any, args: any) => {
      ctx.db.edges.insert(args);
      logOperation(ctx, "add_edge", args);
    },
  },

  remove_edge: {
    args: { id: t.string() },
    handler: (ctx: any, { id }: any) => {
      ctx.db.edges.id.delete(id);
      logOperation(ctx, "remove_edge", { id });
    },
  },

  update_node_layout: {
    args: {
      height: t.f32(),
      id: t.string(),
      width: t.f32(),
      x: t.f32(),
      y: t.f32(),
    },
    handler: (ctx: any, { height, id, width, x, y }: any) => {
      const node = ctx.db.nodes.id.find(id);
      if (node) {
        ctx.db.nodes.id.update({ ...node, height, posX: x, posY: y, width });
        logOperation(ctx, "update_node_layout", { height, id, width, x, y });
      }
    },
  },

  update_viewport: {
    args: { id: t.string(), x: t.f32(), y: t.f32(), zoom: t.f32() },
    handler: (ctx: any, args: any) => {
      const state = ctx.db.viewportState.id.find(args.id);
      if (state) {
        ctx.db.viewportState.id.update(args);
      } else {
        ctx.db.viewportState.insert(args);
      }
    },
  },

  update_widget_value: {
    args: { nodeId: t.string(), valueJson: t.string(), widgetId: t.string() },
    handler: (ctx: any, { nodeId, valueJson, widgetId }: any) => {
      const id = `${nodeId}:${widgetId}`;
      const existing = ctx.db.widgetValues.id.find(id);
      if (existing) {
        ctx.db.widgetValues.id.update({ ...existing, valueJson });
      } else {
        ctx.db.widgetValues.insert({ id, nodeId, valueJson, widgetId });
      }
      logOperation(ctx, "update_widget_value", { nodeId, valueJson, widgetId });
    },
  },
};