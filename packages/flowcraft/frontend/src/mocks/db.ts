import type { AppNode, Edge } from "../types";
import type { Viewport } from "@xyflow/react";

export let serverVersion = 0;

export const incrementVersion = () => {
  serverVersion++;
};

export const serverGraph: {
  nodes: AppNode[];
  edges: Edge[];
  viewport?: Viewport;
} = {
  nodes: [],
  edges: [],
  viewport: { x: 0, y: 0, zoom: 0.7 },
};

export const setServerNodes = (nodes: AppNode[]) => {
  serverGraph.nodes = nodes;
};

export const setServerEdges = (edges: Edge[]) => {
  serverGraph.edges = edges;
};
