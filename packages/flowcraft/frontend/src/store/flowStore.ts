import { create as createProto } from "@bufbuild/protobuf";
import { applyEdgeChanges, applyNodeChanges, type Edge as RFEdge } from "@xyflow/react";
import { create } from "zustand";

import { PositionSchema } from "@/generated/flowcraft/v1/core/base_pb";
import { EdgeSchema } from "@/generated/flowcraft/v1/core/node_pb";
import { type WidgetSignal } from "@/generated/flowcraft/v1/core/signals_pb";
import { type AppNode, MutationSource, Scope } from "@/types";
import { globalToLocal, localToGlobal } from "@/utils/coordinateUtils";
import { socketClient } from "@/utils/SocketClient";

import { commit } from "./orchestrator";
import { type RFState } from "./types";
import { computeView } from "./viewLogic";

export const useFlowStore = create<RFState>((set, get) => ({
  activeScopeId: null,
  addNode: (node) => {
    commit(
      (draft) => {
        draft.nodesById[node.id] = node;
      },
      { description: `Added node ${node.id}` },
    );
  },
  clipboard: null,
  dispatchNodeEvent: (type, payload) => {
    set({ lastNodeEvent: { payload, timestamp: Date.now(), type } });
  },
  edges: [],
  edgesById: {},
  handleIncomingWidgetSignal: (payload) => {
    console.debug("[Store] Incoming widget signal", payload);
  },
  lastInboundChange: null,
  lastLocalUpdate: {},
  lastNodeEvent: null,
  moveNodeToScope: (nodeId, newScopeId) => {
    commit(
      (draft) => {
        const n = draft.nodesById[nodeId];
        if (n) {
          n.scopeId = newScopeId ?? Scope.ROOT;
          if (n.presentation) {
            n.presentation.scopeId = newScopeId ?? Scope.ROOT;
          }
        }
      },
      { description: `Moved node ${nodeId} to scope ${newScopeId}` },
    );
  },
  nodeRelations: {},
  nodes: [],
  nodesById: {},

  onConnect: (connection) => {
    const protoEdge = createProto(EdgeSchema, {
      edgeId: crypto.randomUUID(),
      sourceHandle: connection.sourceHandle ?? "",
      sourceNodeId: connection.source,
      targetHandle: connection.targetHandle ?? "",
      targetNodeId: connection.target,
    });

    commit(
      (draft) => {
        const edge: RFEdge = {
          id: protoEdge.edgeId,
          source: protoEdge.sourceNodeId,
          sourceHandle: protoEdge.sourceHandle,
          target: protoEdge.targetNodeId,
          targetHandle: protoEdge.targetHandle,
        };
        draft.edgesById[edge.id] = edge;
      },
      { description: "Connected nodes" },
    );
  },

  onEdgesChange: (changes) => {
    const hasRemoval = changes.some((c) => c.type === "remove");
    if (hasRemoval) {
      commit(
        (draft) => {
          const currentEdges = Object.values(draft.edgesById);
          const updatedEdges = applyEdgeChanges(changes, currentEdges);
          const nextIds = new Set(updatedEdges.map((e) => e.id));

          Object.keys(draft.edgesById).forEach((id) => {
            if (!nextIds.has(id)) delete draft.edgesById[id];
          });

          updatedEdges.forEach((e) => {
            draft.edgesById[e.id] = e;
          });
        },
        { description: "Edges removed", source: MutationSource.SOURCE_USER },
      );
    } else {
      set((state) => {
        const currentEdges = Object.values(state.edgesById);
        const updatedEdges = applyEdgeChanges(changes, currentEdges);
        const nextEdgesById: Record<string, RFEdge> = {};
        updatedEdges.forEach((e) => {
          nextEdgesById[e.id] = e;
        });
        return { edgesById: nextEdgesById };
      });
      get().refreshView();
    }
  },

  onNodesChange: (changes) => {
    const isInteractionEnd = changes.some(
      (c) => (c.type === "position" && !c.dragging) || (c.type === "dimensions" && !c.resizing),
    );
    const hasStructureChanges = changes.some((c) => c.type === "remove");
    const hasDataChanges = changes.some((c) => c.type === "position" || c.type === "dimensions");

    if (hasStructureChanges || (hasDataChanges && isInteractionEnd)) {
      commit(
        (draft) => {
          const currentNodes = Object.values(draft.nodesById);
          const updatedNodes = applyNodeChanges(changes, currentNodes);
          const nextIds = new Set(updatedNodes.map((n) => n.id));

          // 1. Handle Removals
          Object.keys(draft.nodesById).forEach((id) => {
            if (!nextIds.has(id)) delete draft.nodesById[id];
          });

          // 2. Handle Updates/Adds
          updatedNodes.forEach((n) => {
            const dn = draft.nodesById[n.id];
            if (dn) {
              // Merge React Flow properties
              Object.assign(dn, n);

              // Sync presentation state for persistence
              if (dn.presentation) {
                if (dn.presentation.position) {
                  dn.presentation.position.x = n.position.x;
                  dn.presentation.position.y = n.position.y;
                } else {
                  dn.presentation.position = createProto(PositionSchema, { x: n.position.x, y: n.position.y });
                }
                dn.presentation.width = n.width ?? dn.presentation.width;
                dn.presentation.height = n.height ?? dn.presentation.height;
              }
            } else {
              draft.nodesById[n.id] = n;
            }
          });
        },
        {
          description: hasStructureChanges ? "Nodes removed" : "Interaction commit",
          isInteractionEnd: true,
          source: MutationSource.SOURCE_USER,
        },
      );
    } else {
      // Intermediate updates (dragging/resizing) - performance optimized bypass
      set((state) => {
        const nextNodes = applyNodeChanges(changes, Object.values(state.nodesById));
        const nextNodesById = { ...state.nodesById };
        nextNodes.forEach((n) => {
          const existing = nextNodesById[n.id];
          if (existing) {
            nextNodesById[n.id] = { ...existing, ...n };
          }
        });
        return { nodesById: nextNodesById };
      });
      get().refreshView();
    }
  },

  redo: () => {
    const { edgesById, nodesById, redoStack, undoStack } = get();
    const next = redoStack[0];
    if (!next) return;

    const current = structuredClone({ edgesById, nodesById });
    set({
      edgesById: next.edgesById,
      nodesById: next.nodesById,
      redoStack: redoStack.slice(1),
      undoStack: [current, ...undoStack],
    });
    get().refreshView();
  },

  redoStack: [],

  refreshView: () => {
    const { edgesById, nodesById } = get();
    const { edges, nodeRelations, nodes } = computeView(nodesById, edgesById);
    set({ edges, nodeRelations, nodes });
  },

  reparentNode: (nodeId, newParentId) => {
    const { nodesById } = get();
    const node = nodesById[nodeId];
    if (!node) return;

    // Reparenting now only changes the physical parentId within the SAME scope
    // Cross-scope movement is a separate "moveNodeToScope" operation
    const currentGlobalPos = localToGlobal(node.position, node.parentId ?? null, nodesById);
    const newLocalPos = globalToLocal(currentGlobalPos, newParentId, nodesById);

    commit(
      (draft) => {
        const n = draft.nodesById[nodeId];
        if (n) {
          n.parentId = newParentId ?? undefined;
          n.position = newLocalPos;
          if (n.presentation) {
            n.presentation.parentId = newParentId ?? "";
          }
        }
      },
      { description: `Reparent node ${nodeId}` },
    );
  },

  resetStore: () => {
    set({ edges: [], edgesById: {}, nodes: [], nodesById: {}, redoStack: [], undoStack: [] });
  },

  sendNodeSignal: (signal) => {
    const conn = get().spacetimeConn;
    if (conn) {
      conn.pbreducers.sendNodeSignal({ signal });
    }
  },

  sendWidgetSignal: (signal: unknown) => {
    socketClient.send({
      payload: {
        case: "widgetSignal",
        value: signal as WidgetSignal,
      },
    });
  },
  setClipboard: (content) => {
    set({ clipboard: content });
  },

  setEdges: (edges) => {
    const edgesById: Record<string, RFEdge> = {};
    edges.forEach((e) => (edgesById[e.id] = e));
    set({ edgesById });
    get().refreshView();
  },

  setGraph: (g) => {
    const nodesById: Record<string, AppNode> = {};
    const edgesById: Record<string, RFEdge> = {};
    g.nodes.forEach((n) => (nodesById[n.id] = n));
    g.edges.forEach((e) => (edgesById[e.id] = e));
    set({ edgesById, nodesById });
    get().refreshView();
  },

  setNodes: (nodes) => {
    const nodesById: Record<string, AppNode> = {};
    nodes.forEach((n) => (nodesById[n.id] = n));
    set({ nodesById });
    get().refreshView();
  },

  setSpacetimeConn: (conn) => {
    set({ spacetimeConn: conn });
  },

  spacetimeConn: null,
  takeSnapshot: () => {
    const { edgesById, nodesById, undoStack } = get();
    const snapshot = structuredClone({ edgesById, nodesById });
    set({
      redoStack: [],
      undoStack: [snapshot, ...undoStack].slice(0, 50),
    });
  },
  undo: () => {
    const { edgesById, nodesById, redoStack, undoStack } = get();
    const previous = undoStack[0];
    if (!previous) return;

    const current = structuredClone({ edgesById, nodesById });
    set({
      edgesById: previous.edgesById,
      nodesById: previous.nodesById,
      redoStack: [current, ...redoStack],
      undoStack: undoStack.slice(1),
    });
    get().refreshView();
  },
  undoStack: [],
  viewport: { x: 0, y: 0, zoom: 1 },
}));
