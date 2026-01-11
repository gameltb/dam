import { t, table } from "spacetimedb/server";

export const nodes = table(
  {
    name: "nodes",
    public: true,
  },
  {
    dataJson: t.string(),
    height: t.f32(),
    id: t.string().primaryKey(),
    isSelected: t.bool(),
    kind: t.u32(),
    parentId: t.string(),
    posX: t.f32(),
    posY: t.f32(),
    templateId: t.string(),
    width: t.f32(),
  },
);

export const edges = table(
  {
    name: "edges",
    public: true,
  },
  {
    id: t.string().primaryKey(),
    sourceHandle: t.string(),
    sourceId: t.string(),
    targetHandle: t.string(),
    targetId: t.string(),
  },
);

export const viewportState = table(
  {
    name: "viewport_state",
    public: true,
  },
  {
    id: t.string().primaryKey(),
    x: t.f32(),
    y: t.f32(),
    zoom: t.f32(),
  },
);
