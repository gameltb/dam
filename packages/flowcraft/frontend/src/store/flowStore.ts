import { create as createProto } from "@bufbuild/protobuf";
import { create } from "zustand";

import { PositionSchema } from "@/generated/flowcraft/v1/core/base_pb";
import { EdgeSchema } from "@/generated/flowcraft/v1/core/node_pb";
import { type WidgetSignal } from "@/generated/flowcraft/v1/core/signals_pb";
import { type AppEdge, type AppNode, MutationSource, Scope } from "@/types";
import { globalToLocal, localToGlobal } from "@/utils/coordinateUtils";
import { GraphMapper } from "@/utils/graphMapper";
import { log } from "@/utils/logger";
import { sanitizeNode } from "@/utils/nodeUtils";
import { socketClient } from "@/utils/SocketClient";

import { commit } from "./orchestrator";
import { type RFState } from "./types";
import { computeView } from "./viewLogic";

export const useFlowStore = create<RFState>((set, get) => ({
  activeGraphId: "default",
  activeScopeId: null,
  addNode: (node) => {
    const { activeGraphId } = get();
    commit(
      (draft) => {
        const newNode = { ...node, graphId: activeGraphId || "default" };
        draft.nodesById[node.id] = newNode;
      },
      { description: `Added node ${node.id}` },
    );
  },
  clipboard: null,
  commitNodes: (nodes) => {
    log.debug("Store", `Committing persistent transform for ${nodes.length} nodes`);
    commit(
      (draft) => {
        nodes.forEach((n) => {
          const dn = draft.nodesById[n.id];
          if (dn) {
            // 1. Precise sync of UI properties ONLY
            if (n.position) {
              dn.position = { x: n.position.x, y: n.position.y };
            }
            if (n.width) dn.width = n.width;
            if (n.height) dn.height = n.height;
            if (n.measured) dn.measured = { ...n.measured };
            dn.selected = !!n.selected;
            dn.dragging = !!n.dragging;
            dn.resizing = !!n.resizing;

            // 2. Sync presentation state for persistence (Atomic updates to PB nested message)
            if (dn.presentation) {
              const nextPos = n.position || dn.position;
              dn.presentation.position ??= createProto(PositionSchema, { x: 0, y: 0 });
              dn.presentation.position.x = nextPos.x;
              dn.presentation.position.y = nextPos.y;
              if (n.width) dn.presentation.width = n.width;
              if (n.height) dn.presentation.height = n.height;
            }
          }
        });
      },
      { description: "Interaction commit", isInteractionEnd: true, source: MutationSource.SOURCE_USER },
    );
  },
  deleteNodes: (ids) => {
    log.debug("Store", `Deleting nodes: ${ids.join(", ")}`);
    commit(
      (draft) => {
        ids.forEach((id) => {
          delete draft.nodesById[id];
        });
      },
      { description: "Nodes removed", source: MutationSource.SOURCE_USER },
    );
  },
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
    const { activeGraphId } = get();
    const protoEdge = createProto(EdgeSchema, {
      edgeId: crypto.randomUUID(),
      graphId: activeGraphId || "default",
      sourceHandle: connection.sourceHandle ?? "",
      sourceNodeId: connection.source,
      targetHandle: connection.targetHandle ?? "",
      targetNodeId: connection.target,
    });

    commit(
      (draft) => {
        const edge: AppEdge = {
          graphId: activeGraphId || "default",
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

  onEdgesChange: (_changes) => {
    // Uncontrolled: React Flow handles internal edge state.
    // Side effects (like removal) are handled via granular event hooks.
  },

  onNodesChange: (_changes) => {
    // Uncontrolled: React Flow handles internal node positions and selection.
    // Business logic is triggered onDragStop or onNodesDelete.
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

  refreshView: (options) => {
    const { edgesById, nodes, nodeRelations, nodesById } = get();
    const existingRelations = options?.skipLayout ? nodeRelations : undefined;
    const {
      edges,
      nodeRelations: nextRelations,
      nodes: nextNodes,
    } = computeView(nodesById, edgesById, existingRelations);

    // Metadata Preservation: Ensure that transient UI state (measured size, interaction flags)
    // is carried over from the current view to the new view array.
    const reconciledNodes = nextNodes.map((n) => {
      const current = nodes.find((vn) => vn.id === n.id);
      if (current) {
        const isInteracting = current.dragging || (current as any).resizing;
        return {
          ...n,
          dragging: current.dragging,
          measured: current.measured,
          position: isInteracting ? current.position : n.position,
          resizing: (current as any).resizing,
          selected: current.selected,
        };
      }
      return n;
    });

    set({ edges: reconciledNodes.length > 0 ? (edges as AppEdge[]) : [], nodeRelations: nextRelations, nodes: reconciledNodes });
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

  setActiveGraph: (id) => {
    set({ activeGraphId: id, activeScopeId: null, edgesById: {}, nodesById: {} });
    get().syncWithDatabase();
  },

  setClipboard: (content) => {
    set({ clipboard: content });
  },
  setEdges: (edges) => {
    const edgesById: Record<string, AppEdge> = {};
    edges.forEach((e) => (edgesById[e.id] = e as AppEdge));
    set({ edgesById });
    get().refreshView();
  },

  setGraph: (g) => {
    const nodesById: Record<string, AppNode> = {};
    const edgesById: Record<string, AppEdge> = {};
    g.nodes.forEach((n) => {
      nodesById[n.id] = sanitizeNode(n);
    });
    g.edges.forEach((e) => (edgesById[e.id] = e as AppEdge));
    set({ edgesById, nodesById });
    get().refreshView();
  },

  setNodes: (nodes) => {
    const nodesById: Record<string, AppNode> = {};
    nodes.forEach((n) => {
      nodesById[n.id] = sanitizeNode(n);
    });
    set({ nodesById });
    get().refreshView();
  },

  setSpacetimeConn: (conn) => {
    set({ spacetimeConn: conn });
  },

  spacetimeConn: null,

  syncWithDatabase: () => {
    const { activeGraphId, spacetimeConn: conn } = get();
    if (!conn) return;

    const nextNodesById = { ...get().nodesById };
    const nextEdgesById: Record<string, AppEdge> = {};

    // 1. Sync Base Nodes (Filtered by graphId)
    for (const row of conn.db.nodes.iter()) {
      if (row.graphId === activeGraphId) {
        if (!nextNodesById[row.nodeId]) {
          nextNodesById[row.nodeId] = GraphMapper.createSkeleton(row);
        }
      }
    }

    // 2. Sync Edges (Filtered by graphId)
    for (const row of conn.db.edges.iter()) {
      if (row.graphId === activeGraphId) {
        const edge = GraphMapper.toEdge(row);
        nextEdgesById[edge.id] = edge;
      }
    }

    // 3. Sync Transforms
    for (const row of conn.db.nodeTransforms.iter()) {
      const n = nextNodesById[row.nodeId];
      if (n) nextNodesById[row.nodeId] = GraphMapper.applyTransform(n, row);
    }

    // 4. Sync Metadata
    for (const row of conn.db.nodeMetadata.iter()) {
      const n = nextNodesById[row.nodeId];
      if (n) nextNodesById[row.nodeId] = GraphMapper.applyMetadata(n, row);
    }

    // 5. Sync Data
    for (const row of conn.db.nodeData.iter()) {
      const n = nextNodesById[row.nodeId];
      if (n) nextNodesById[row.nodeId] = GraphMapper.applyData(n, row);
    }

    set({ edgesById: nextEdgesById, nodesById: nextNodesById });
    get().refreshView();
  },
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
