import { create as createProto, fromJson } from "@bufbuild/protobuf";
import { ValueSchema } from "@bufbuild/protobuf/wkt";
import { applyEdgeChanges, applyNodeChanges, type Edge } from "@xyflow/react";
import { create } from "zustand";

import { NodeSchema } from "@/generated/flowcraft/v1/core/node_pb";
import {
  AddNodeRequestSchema,
  PathUpdateRequest_UpdateType,
  PathUpdateRequestSchema,
  ReparentNodeRequestSchema,
} from "@/generated/flowcraft/v1/core/service_pb";
import { type AppNode, FlowEvent, MutationSource } from "@/types";
import { globalToLocal, localToGlobal } from "@/utils/coordinateUtils";
import { createNodeDraft, type Draftable, Err, type Result } from "@/utils/draft";
import { calculateInverse, getFriendlyDescription } from "@/utils/historyUtils";
import { appNodeToProto } from "@/utils/nodeProtoUtils";

import { type GraphMutationEvent, MutationDirection } from "./middleware/types";
import { handleMutation } from "./mutationHandlers";
import { NotificationType, useNotificationStore } from "./notificationStore";
import { initStoreOrchestrator } from "./orchestrator";
import { dispatchToSpacetime } from "./spacetimeDispatcher";
import { type HistoryEntry, type MutationInput, type RFState } from "./types";
import { useUiStore } from "./uiStore";

export const useFlowStore = create<RFState>()((set, get) => ({
  addNode: (node) => {
    get().applyMutations([createProto(AddNodeRequestSchema, { node: appNodeToProto(node) })]);
  },
  allEdges: [],
  allNodes: [],
  applyMutations: (inputs, context) => {
    const source = context?.source ?? MutationSource.SOURCE_USER;
    const direction = source === MutationSource.SOURCE_SYNC ? MutationDirection.INCOMING : MutationDirection.OUTGOING;

    const pipeline = initStoreOrchestrator();
    pipeline.execute({ context: context ?? {}, direction, mutations: inputs }, (finalEvent: GraphMutationEvent) => {
      const currentNodes = [...get().allNodes];
      const currentEdges = [...get().allEdges];

      // 1. 记录历史
      if (direction === MutationDirection.OUTGOING && !context?.isHistoryOp) {
        const inverses = finalEvent.mutations
          .map((m: MutationInput) => calculateInverse(m, currentNodes))
          .filter((m: MutationInput | null): m is MutationInput => m !== null);

        if (inverses.length > 0) {
          const entry: HistoryEntry = {
            description:
              context?.description ||
              (finalEvent.mutations[0] ? getFriendlyDescription(finalEvent.mutations[0]) : "Unknown Operation"),
            forward: [...finalEvent.mutations],
            id: crypto.randomUUID(),
            inverse: inverses,
            scopeId: useUiStore.getState().activeScopeId,
            timestamp: Date.now(),
          };
          set((state) => ({
            redoStack: [],
            undoStack: [entry, ...state.undoStack].slice(0, 50),
          }));
        }
      }

      // 2. 执行状态更新
      let nextNodes = currentNodes;
      let nextEdges = currentEdges;
      finalEvent.mutations.forEach((mutInput: MutationInput) => {
        const result = handleMutation(mutInput, nextNodes, nextEdges);
        nextNodes = result.nodes;
        nextEdges = result.edges;
      });

      set({ allEdges: nextEdges, allNodes: nextNodes });
      get().refreshView();

      // 3. 后端同步
      if (direction === MutationDirection.OUTGOING && get().spacetimeConn) {
        finalEvent.mutations.forEach((mut: MutationInput) => {
          dispatchToSpacetime(get().spacetimeConn!, mut);
        });
      }
    });
  },
  dispatchNodeEvent: (type: FlowEvent, payload: Record<string, unknown>) => {
    set({ lastNodeEvent: { payload, timestamp: Date.now(), type } });
  },
  edges: [],
  handleIncomingWidgetSignal: () => {},
  lastLocalUpdate: {},

  lastNodeEvent: null,

  nodeDraft: (nodeIdOrNode: AppNode | string): Result<Draftable<AppNode>> => {
    const node = typeof nodeIdOrNode === "string" ? get().allNodes.find((n) => n.id === nodeIdOrNode) : nodeIdOrNode;

    if (!node) return Err(`Node ${String(nodeIdOrNode)} not found`);

    return createNodeDraft(node.id, node, NodeSchema, (path: string, value: unknown) => {
      get().applyMutations([
        createProto(PathUpdateRequestSchema, {
          path: path,
          targetId: node.id,
          type: PathUpdateRequest_UpdateType.REPLACE,
          value: fromJson(ValueSchema, value as any),
        }),
      ]);
    });
  },

  nodes: [],

  onConnect: () => {},

  onEdgesChange: (changes) => {
    set({ allEdges: applyEdgeChanges(changes, get().allEdges) });
    get().refreshView();
  },

  onNodesChange: (changes) => {
    set({ allNodes: applyNodeChanges(changes, get().allNodes) });
    get().refreshView();
  },

  redo: () => {
    const { redoStack, undoStack } = get();
    if (redoStack.length === 0) return;

    const entry = redoStack[0]!;
    const remaining = redoStack.slice(1);

    if (entry.scopeId !== useUiStore.getState().activeScopeId) {
      useUiStore.getState().setActiveScope(entry.scopeId);
    }

    get().applyMutations(entry.forward, {
      description: `重做: ${entry.description}`,
      isHistoryOp: true,
    });

    useNotificationStore
      .getState()
      .addNotification({ message: `已重做: ${entry.description}`, type: NotificationType.INFO });

    set({
      redoStack: remaining,
      undoStack: [entry, ...undoStack],
    });
  },

  redoStack: [],

  refreshView: () => {
    const activeScopeId = useUiStore.getState().activeScopeId;
    const allNodes = get().allNodes;
    const allEdges = get().allEdges;

    const nextNodes = allNodes.filter((n) => (n.parentId || null) === activeScopeId);
    const nextEdges = allEdges.filter((e) => {
      const s = allNodes.find((n) => n.id === e.source);
      const t = allNodes.find((n) => n.id === e.target);
      return (s?.parentId || null) === activeScopeId && (t?.parentId || null) === activeScopeId;
    });

    const currentNodes = get().nodes;
    const currentEdges = get().edges;

    const nodesChanged = nextNodes.length !== currentNodes.length || nextNodes.some((n, i) => n !== currentNodes[i]);
    const edgesChanged = nextEdges.length !== currentEdges.length || nextEdges.some((e, i) => e !== currentEdges[i]);

    if (nodesChanged || edgesChanged) {
      set({ edges: nextEdges, nodes: nextNodes });
    }
  },

  reparentNode: (nodeId, newParentId) => {
    const node = get().allNodes.find((n) => n.id === nodeId);
    if (!node) return;
    const currentGlobalPos = localToGlobal(node.position, node.parentId || null, get().allNodes);
    const newLocalPos = globalToLocal(currentGlobalPos, newParentId, get().allNodes);
    get().applyMutations([
      createProto(ReparentNodeRequestSchema, {
        newParentId: newParentId || "",
        newPosition: newLocalPos,
        nodeId,
      }),
    ]);
  },
  resetStore: () => {
    set({ allEdges: [], allNodes: [], edges: [], nodes: [], redoStack: [], undoStack: [] });
  },
  sendNodeSignal: (signal) => get().spacetimeConn?.pbreducers.sendNodeSignal({ signal }),
  sendWidgetSignal: () => {},
  setEdges: (allEdges: Edge[]) => {
    set({ allEdges });
    get().refreshView();
  },
  setGraph: (g: { edges: Edge[]; nodes: AppNode[] }) => {
    set({ allEdges: g.edges, allNodes: g.nodes });
    get().refreshView();
  },
  setNodes: (allNodes: AppNode[]) => {
    set({ allNodes });
    get().refreshView();
  },
  undo: () => {
    const { redoStack, undoStack } = get();
    if (undoStack.length === 0) return;

    const entry = undoStack[0]!;
    const remaining = undoStack.slice(1);

    if (entry.scopeId !== useUiStore.getState().activeScopeId) {
      useUiStore.getState().setActiveScope(entry.scopeId);
    }

    get().applyMutations(entry.inverse, {
      description: `撤销: ${entry.description}`,
      isHistoryOp: true,
    });

    useNotificationStore
      .getState()
      .addNotification({ message: `已撤销: ${entry.description}`, type: NotificationType.INFO });

    set({
      redoStack: [entry, ...redoStack],
      undoStack: remaining,
    });
  },
  undoStack: [],
}));
