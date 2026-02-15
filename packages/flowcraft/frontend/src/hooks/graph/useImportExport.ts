import { create as createProto, toJsonString } from "@bufbuild/protobuf";

import { ChatMessagePartSchema } from "@/generated/flowcraft/v1/actions/chat_actions_pb";
import { NotificationType, useNotificationStore } from "@/store/notificationStore";
import { commit } from "@/store/orchestrator";
import { useFlowStore } from "@/store/flowStore";
import { type AppEdge, type AppNode, AppNodeType, type DynamicNodeData } from "@/types";
import { log } from "@/utils/logger";

import { useLayoutOperations } from "./useLayoutOperations";

const LAYOUT = {
  LAYOUT_SETTLE_DELAY_MS: 100,
  MESSAGE_HORIZONTAL_SPACING: 350,
  MESSAGE_NODE_HEIGHT: 150,
  MESSAGE_NODE_WIDTH: 300,
  MESSAGE_VERTICAL_OFFSET: 300,
  MESSAGE_VERTICAL_SPACING: 250,
  MESSAGES_PER_ROW: 5,
  SESSION_HORIZONTAL_SPACING: 1000,
  SESSION_ROOT_HEIGHT: 200,
  SESSION_ROOT_WIDTH: 300,
} as const;

interface ImportedMessage {
  content?: string;
  id: string;
  parent?: string;
  role?: string;
  timestamp?: number;
  type?: string;
}

interface ImportedSession {
  conv: { currNode?: string; id?: string; name?: string };
  messages: unknown[];
}

interface ParsedGraph {
  edges?: unknown[];
  nodes?: unknown[];
}

function calculateMessagePosition(sessionIndex: number, messageIndex: number) {
  return {
    x:
      sessionIndex * LAYOUT.SESSION_HORIZONTAL_SPACING +
      (messageIndex % LAYOUT.MESSAGES_PER_ROW) * LAYOUT.MESSAGE_HORIZONTAL_SPACING,
    y:
      LAYOUT.MESSAGE_VERTICAL_OFFSET +
      Math.floor(messageIndex / LAYOUT.MESSAGES_PER_ROW) * LAYOUT.MESSAGE_VERTICAL_SPACING,
  };
}

function createChatNode(session: ImportedSession, sessionIndex: number, graphId: string): AppNode {
  const sessionRootId = session.conv.id ?? `session-${String(sessionIndex)}`;
  return {
    data: {
      activeMode: 1,
      availableModes: [1, 2, 3],
      displayName: session.conv.name ?? "Imported Session",
      extension: {
        case: "chat",
        value: {
          conversationHeadId: session.conv.currNode ?? "",
          isHistoryCleared: false,
          treeId: sessionRootId,
        },
      },
    } as DynamicNodeData,
    graphId,
    height: LAYOUT.SESSION_ROOT_HEIGHT,
    id: sessionRootId,
    position: { x: sessionIndex * LAYOUT.SESSION_HORIZONTAL_SPACING, y: 0 },
    scopeId: "root",
    type: AppNodeType.DYNAMIC,
    width: LAYOUT.SESSION_ROOT_WIDTH,
  };
}

function createEdge(sourceId: string, targetId: string, graphId: string): AppEdge {
  return {
    graphId,
    id: `e-${sourceId}-${targetId}`,
    source: sourceId,
    target: targetId,
    type: "default",
  };
}

function createMessageNode(msg: ImportedMessage, position: { x: number; y: number }, sessionRootId: string, graphId: string): AppNode {
  const partsJson = JSON.stringify([
    JSON.parse(
      toJsonString(
        ChatMessagePartSchema,
        createProto(ChatMessagePartSchema, {
          part: { case: "text", value: msg.content ?? "" },
        }),
      ),
    ),
  ]);

  return {
    data: {
      metadata: {
        parts_json: partsJson,
        role: msg.role,
        timestamp: msg.timestamp?.toString(),
      },
    } as unknown as DynamicNodeData,
    graphId,
    height: LAYOUT.MESSAGE_NODE_HEIGHT,
    id: msg.id,
    parentId: sessionRootId,
    position,
    scopeId: "root",
    type: AppNodeType.CHAT_MESSAGE,
    width: LAYOUT.MESSAGE_NODE_WIDTH,
  };
}

function isValidSession(data: unknown): data is ImportedSession {
  if (typeof data !== "object" || data === null) return false;
  const session = data as Record<string, unknown>;
  return (
    "conv" in session && typeof session.conv === "object" && "messages" in session && Array.isArray(session.messages)
  );
}

function parseMessage(msg: unknown): ImportedMessage | null {
  if (typeof msg !== "object" || msg === null) return null;
  const m = msg as Record<string, unknown>;
  if (typeof m.id !== "string") return null;
  return {
    content: typeof m.content === "string" ? m.content : undefined,
    id: m.id,
    parent: typeof m.parent === "string" ? m.parent : undefined,
    role: typeof m.role === "string" ? m.role : undefined,
    timestamp: typeof m.timestamp === "number" ? m.timestamp : undefined,
    type: typeof m.type === "string" ? m.type : undefined,
  };
}

export const useImportExport = () => {
  const { autoLayout } = useLayoutOperations();
  const addNotification = useNotificationStore((s) => s.addNotification);
  const activeGraphId = useFlowStore((s) => s.activeGraphId);

  const importGraph = (data: string) => {
    let parsed: ParsedGraph;
    try {
      parsed = JSON.parse(data) as ParsedGraph;
    } catch (err) {
      log.error("useImportExport/importGraph", err);
      addNotification({ message: "Failed to import graph: invalid JSON format", type: NotificationType.ERROR });
      return;
    }

    const graphId = activeGraphId || "default";

    commit(
      (draft) => {
        // Note: During import, we don't clear nodesById anymore, 
        // as we want to support partial imports into the active graph.
        
        parsed.nodes?.forEach((n: unknown) => {
          const node = n as any;
          if (node.id) {
            draft.nodesById[node.id] = { ...node, graphId } as AppNode;
          }
        });
        parsed.edges?.forEach((e: unknown) => {
          const edge = e as any;
          if (edge.id) {
            draft.edgesById[edge.id] = { ...edge, graphId } as AppEdge;
          }
        });
      },
      { description: "Import graph from JSON", isHistoryOp: true },
    );
    addNotification({ message: "Graph imported successfully", type: NotificationType.SUCCESS });
  };

  const importConversations = (data: string) => {
    let sessions: unknown[];
    try {
      sessions = JSON.parse(data) as unknown[];
    } catch (err) {
      log.error("useImportExport/importConversations/parse", err);
      addNotification({ message: "Failed to parse conversation data: invalid JSON", type: NotificationType.ERROR });
      return;
    }

    if (!Array.isArray(sessions)) {
      addNotification({ message: "Invalid conversation format: expected an array", type: NotificationType.ERROR });
      return;
    }

    const validSessions = sessions.filter(isValidSession);
    if (validSessions.length === 0) {
      addNotification({ message: "No valid sessions found in imported data", type: NotificationType.INFO });
      return;
    }

    if (validSessions.length < sessions.length) {
      const skippedCount = sessions.length - validSessions.length;
      log.warn("useImportExport/importConversations", `${String(skippedCount)} invalid sessions skipped`);
    }

    const graphId = activeGraphId || "default";

    try {
      commit(
        (draft) => {
          validSessions.forEach((session, sIdx) => {
            const chatNode = createChatNode(session, sIdx, graphId);
            draft.nodesById[chatNode.id] = chatNode;

            const msgNodes: Record<string, AppNode> = {};
            session.messages.forEach((msgData, mIdx) => {
              const msg = parseMessage(msgData);
              if (!msg) return;

              const position = calculateMessagePosition(sIdx, mIdx);
              const node = createMessageNode(msg, position, chatNode.id, graphId);
              msgNodes[node.id] = node;
              draft.nodesById[node.id] = node;
            });

            session.messages.forEach((msgData) => {
              const msg = parseMessage(msgData);
              if (!msg) return;

              if (msg.parent && msgNodes[msg.parent]) {
                draft.edgesById[`e-${msg.parent}-${msg.id}`] = createEdge(msg.parent, msg.id, graphId);
              } else if (msg.type === "root" || !msg.parent) {
                draft.edgesById[`e-${chatNode.id}-${msg.id}`] = createEdge(chatNode.id, msg.id, graphId);
              }
            });
          });
        },
        { description: "Imported message tree subgraphs", isHistoryOp: true },
      );

      const sessionCount = validSessions.length;
      addNotification({
        message: `Imported ${String(sessionCount)} session(s) successfully`,
        type: NotificationType.SUCCESS,
      });

      setTimeout(() => {
        autoLayout();
      }, LAYOUT.LAYOUT_SETTLE_DELAY_MS);
    } catch (err) {
      log.error("useImportExport/importConversations", err);
      addNotification({ message: "Failed to import conversations: unexpected error", type: NotificationType.ERROR });
    }
  };

  return { autoLayout, importConversations, importGraph };
};
