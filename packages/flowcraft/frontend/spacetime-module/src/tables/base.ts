import { t, table } from "spacetimedb/server";

import { Edge, Node, Viewport } from "../generated/generated_schema";

export const nodes = table(
  {
    name: "nodes",
    public: true,
  },
  {
    nodeId: t.string().primaryKey(),
    state: Node,
  },
);

export const edges = table(
  {
    name: "edges",
    public: true,
  },
  {
    edgeId: t.string().primaryKey(),
    state: Edge,
  },
);

export const viewportState = table(
  {
    name: "viewport_state",
    public: true,
  },
  {
    id: t.string().primaryKey(), // "default"
    state: Viewport,
  },
);
