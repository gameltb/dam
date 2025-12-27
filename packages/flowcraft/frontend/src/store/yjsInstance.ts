import * as Y from "yjs";

// Global Yjs document instance
export const ydoc = new Y.Doc();

// Shared types for nodes and edges
export const yNodes = ydoc.getMap<unknown>("nodes");
export const yEdges = ydoc.getMap<unknown>("edges");
