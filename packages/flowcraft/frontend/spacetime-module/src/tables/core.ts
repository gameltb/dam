import { t, table } from "spacetimedb/server";

export const graphs = table(
  {
    name: "graphs",
    public: true,
  },
  {
    graphId: t.string().primaryKey(),
    name: t.string(),
    ownerId: t.string().index(),
    createdAt: t.u64(),
  },
);

export const nodes = table(
  {
    name: "nodes",
    public: true,
  },
  {
    graphId: t.string().index(),
    nodeId: t.string().primaryKey(),
    nodeKind: t.u32(),
    templateId: t.string().index(),
  },
);

export const edges = table(
  {
    name: "edges",
    public: true,
  },
  {
    edgeId: t.string().primaryKey(),
    graphId: t.string().index(),
    sourceNodeId: t.string().index(),
    state: Object.assign(t.byteArray(), { __pb_schema: "Edge" }),
    targetNodeId: t.string().index(),
  },
);

export const nodeTransforms = table(
  {
    name: "node_transforms",
    public: true,
  },
  {
    height: t.f32(),
    nodeId: t.string().primaryKey(),
    width: t.f32(),
    x: t.f32(),
    y: t.f32(),
  },
);

export const nodeMetadata = table(
  {
    name: "node_metadata",
    public: true,
  },
  {
    displayName: t.string(),
    graphId: t.string().index(),
    nodeId: t.string().primaryKey(),
    parentId: t.string().index(), // Use empty string for no parent
    scopeId: t.string().index(),
  },
);

export const nodeData = table(
  {
    name: "node_data",
    public: true,
  },
  {
    nodeId: t.string().primaryKey(),
    state: Object.assign(t.byteArray(), { __pb_schema: "NodeData" }),
  },
);

export const viewportState = table(
  {
    name: "viewport_state",
    public: true,
  },
  {
    graphId: t.string().index(),
    id: t.string().primaryKey(), // id here is usually the scopeId
    state: Object.assign(t.byteArray(), { __pb_schema: "Viewport" }),
  },
);

export const widgetValues = table(
  {
    name: "widget_values",
    public: true,
  },
  {
    id: t.string().primaryKey(),
    nodeId: t.string(),
    value: t.string(),
    widgetId: t.string(),
  },
);

export const nodeUiState = table(
  {
    name: "node_ui_state",
    public: true,
  },
  {
    lastUpdated: t.u64(),
    nodeId: t.string().primaryKey(),
    stateJson: t.string(), // Flexible JSON for all transient UI flags
  },
);
