import { create as createProto, fromJson } from "@bufbuild/protobuf";
import { toJson } from "@bufbuild/protobuf";
import { ValueSchema } from "@bufbuild/protobuf/wkt";
import { useReactFlow } from "@xyflow/react";
import { useCallback, useEffect, useRef } from "react";
import { useSpacetimeDB, useTable } from "spacetimedb/react";

import { PresentationSchema } from "@/generated/flowcraft/v1/core/base_pb";
import { NodeDataSchema } from "@/generated/flowcraft/v1/core/node_pb";
import {
  AddNodeRequestSchema,
  PathUpdateRequest_UpdateType,
  PathUpdateRequestSchema,
  RemoveNodeRequestSchema,
} from "@/generated/flowcraft/v1/core/service_pb";
import { DbConnection, tables } from "@/generated/spacetime";
import { ChatStreamStatus, useChatStore } from "@/store/chatStore";
import { useFlowStore } from "@/store/flowStore";
import { useTaskStore } from "@/store/taskStore";
import { MutationSource, TaskStatus, TaskType } from "@/types";
import { convertStdbToPb, wrapReducers } from "@/utils/pb-client";

const safeStringify = (obj: any) => JSON.stringify(obj, (_, v) => (typeof v === "bigint" ? v.toString() : v));

export const useSpacetimeSync = () => {
  const stdb = useSpacetimeDB();
  const { isActive } = stdb;

  const getConnection = useCallback(() => stdb.getConnection<DbConnection>(), [stdb]);

  const [stNodes] = useTable(tables.nodes);
  const [stEdges] = useTable(tables.edges);
  const [stTasks] = useTable(tables.tasks);
  const [stChatMessages] = useTable(tables.chatMessages);
  const [stChatStreams] = useTable(tables.chatStreams);

  const { setViewport } = useReactFlow();
  const updateTask = useTaskStore((state) => state.updateTask);
  const registerTask = useTaskStore((state) => state.registerTask);
  const setChatMessages = useChatStore((state) => state.setMessages);
  const setChatStreams = useChatStore((state) => state.setStreams);

  const viewportInitializedRef = useRef(false);

  const connInitializedRef = useRef<boolean>(false);

  useEffect(() => {
    const conn = getConnection();
    if (conn && isActive && !connInitializedRef.current) {
      const pbConn = wrapReducers(conn);
      useFlowStore.setState({ spacetimeConn: pbConn });
      connInitializedRef.current = true;

      const sessionTaskId = `user-session-${crypto.randomUUID()}`;
      pbConn.reducers.assignCurrentTask({ taskId: sessionTaskId });

      console.log("[Sync] Subscribing to tables...");
      void conn
        .subscriptionBuilder()
        .subscribe([
          "SELECT * FROM nodes",
          "SELECT * FROM edges",
          "SELECT * FROM viewport_state",
          "SELECT * FROM tasks",
          "SELECT * FROM workers",
          "SELECT * FROM task_audit_log",
          "SELECT * FROM chat_messages",
        ]);

      conn.db.chatMessages.onInsert((_, row) => {
        console.log("[Sync] Chat Message INSERTED:", row.id, row.state);
      });

      conn.db.tasks.onDelete((_, row) => {
        console.log(`[Sync] Task DELETED from STDB: ${row.id}, nodeId: ${row.nodeId}`);
        if (row.nodeId) {
          useTaskStore.getState().removeTasksForNode(row.nodeId);
        }
      });

      const onViewportInsert = (_ctx: unknown, row: { id: string; state: unknown }) => {
        if (row.id === "default" && !viewportInitializedRef.current) {
          const viewportState = convertStdbToPb("viewportState", row);
          if (viewportState) {
            if (isNaN(viewportState.x) || isNaN(viewportState.y) || isNaN(viewportState.zoom)) {
              throw new Error("[Sync] Received NaN Viewport from server");
            }
            void setViewport({ x: viewportState.x, y: viewportState.y, zoom: viewportState.zoom });
            viewportInitializedRef.current = true;
          }
        }
      };

      conn.db.viewportState.onInsert(onViewportInsert);
    }
  }, [getConnection, isActive, setViewport]);

  const lastProcessedMessagesRef = useRef<string>("");
  useEffect(() => {
    const messagesJson = safeStringify(stChatMessages);
    if (messagesJson === lastProcessedMessagesRef.current) return;
    lastProcessedMessagesRef.current = messagesJson;

    const pbMessages = stChatMessages
      .map((row) => {
        const pb = convertStdbToPb("chatMessages", row);
        if (pb && !pb.id && row.id) pb.id = row.id;
        return pb;
      })
      .filter(Boolean);
    setChatMessages(pbMessages);
  }, [stChatMessages, setChatMessages]);

  const lastProcessedStreamsRef = useRef<string>("");
  useEffect(() => {
    const streamsJson = safeStringify(stChatStreams);
    if (streamsJson === lastProcessedStreamsRef.current) return;
    lastProcessedStreamsRef.current = streamsJson;

    const streams = stChatStreams.map((row) => ({
      content: row.content,
      nodeId: row.nodeId,
      status: row.status as ChatStreamStatus,
    }));
    setChatStreams(streams);
  }, [stChatStreams, setChatStreams]);

  const lastProcessedTasksRef = useRef<string>("");
  useEffect(() => {
    const tasksJson = safeStringify(stTasks);
    if (tasksJson === lastProcessedTasksRef.current) return;
    lastProcessedTasksRef.current = tasksJson;

    stTasks.forEach((stTask) => {
      const taskStore = useTaskStore.getState();
      const existingTask = taskStore.tasks[stTask.id];
      const statusTag = stTask.status.tag;

      // Map TASK_STATUS_X to TaskStatus.X (e.g. TASK_STATUS_PENDING -> TaskStatus.PENDING)
      // Also handle numeric tags just in case
      const STATUS_NUM_MAP: Record<number, string> = {
        0: "PENDING",
        1: "CLAIMED",
        2: "RUNNING",
        3: "COMPLETED",
        4: "FAILED",
        5: "CANCELLED",
      };

      const statusName =
        typeof statusTag === "number" ? STATUS_NUM_MAP[statusTag] : statusTag.replace("TASK_STATUS_", "");

      const newStatus = TaskStatus[statusName as keyof typeof TaskStatus] as unknown as TaskStatus;
      const currentResult = stTask.result || (newStatus === TaskStatus.PENDING ? "Initializing..." : "");

      if (!existingTask) {
        console.log(
          `[Sync] Registering NEW task: ${stTask.id}, status: ${statusName}, nodeId: ${stTask.nodeId}, result: "${stTask.result}"`,
        );
        registerTask({
          label: `Task: ${stTask.taskType}`,
          message: currentResult,
          nodeId: stTask.nodeId,
          status: newStatus,
          taskId: stTask.id,
          type: TaskType.REMOTE,
        });
      } else if (existingTask.status !== newStatus || existingTask.message !== currentResult) {
        console.log(
          `[Sync] UPDATING task: ${stTask.id}, status: ${statusName}, message changed: ${existingTask.message !== currentResult}, new message: "${currentResult}"`,
        );
        updateTask(stTask.id, {
          message: currentResult,
          status: newStatus,
        });
      }
    });
  }, [stTasks, updateTask, registerTask]);

  useEffect(() => {
    const conn = getConnection();
    if (conn && isActive) {
      conn.db.tasks.onDelete((_, row) => {
        console.log(`[Sync] Task DELETED from STDB: ${row.id}, nodeId: ${row.nodeId}`);
        if (row.nodeId) {
          useTaskStore.getState().removeTasksForNode(row.nodeId);
        }
      });
    }
  }, [getConnection, isActive]);

  const lastProcessedNodesRef = useRef<string>("");
  const processingRemoteUpdateRef = useRef(false);
  useEffect(() => {
    const nodesJson = safeStringify(stNodes);
    const edgesJson = safeStringify(stEdges);
    const combinedJson = nodesJson + edgesJson;

    if (combinedJson === lastProcessedNodesRef.current) return;
    lastProcessedNodesRef.current = combinedJson;

    if (stNodes.length === 0 && stEdges.length === 0) return;
    if (processingRemoteUpdateRef.current) return;

    const { allNodes: localNodes, applyMutations } = useFlowStore.getState();
    const remoteNodeIds = new Set(stNodes.map((n) => n.nodeId));
    const conn = getConnection();

    processingRemoteUpdateRef.current = true;
    try {
      const allMutations: any[] = [];

      stNodes.forEach((nodeRow) => {
        if (!conn) return;
        const node = convertStdbToPb("nodes", nodeRow);
        if (!node) return;

        const localNode = localNodes.find((ln) => ln.id === node.nodeId);

        if (!localNode) {
          allMutations.push(
            createProto(AddNodeRequestSchema, {
              node: {
                nodeId: node.nodeId,
                nodeKind: node.nodeKind,
                presentation: node.presentation,
                state: node.state,
                templateId: node.templateId,
              },
            }),
          );
        } else {
          const { lastLocalUpdate } = useFlowStore.getState();
          const lastUpdate = lastLocalUpdate[node.nodeId] || 0;
          const isRecentlyUpdatedLocally = Date.now() - lastUpdate < 3000;
          const isInteracting = (localNode.dragging ?? false) || (localNode.resizing ?? false) || localNode.selected;

          if (!isInteracting && !isRecentlyUpdatedLocally) {
            // Check for actual changes before applying mutations to prevent loops
            const hasPresentationChanged = safeStringify(localNode.presentation) !== safeStringify(node.presentation);
            const hasStateChanged = safeStringify(localNode.data) !== safeStringify(node.state);

            if (hasPresentationChanged || hasStateChanged) {
              allMutations.push(
                createProto(PathUpdateRequestSchema, {
                  path: "presentation",
                  targetId: node.nodeId,
                  type: PathUpdateRequest_UpdateType.REPLACE,
                  value: fromJson(ValueSchema, toJson(PresentationSchema, node.presentation)),
                }),
              );
              allMutations.push(
                createProto(PathUpdateRequestSchema, {
                  path: "data",
                  targetId: node.nodeId,
                  type: PathUpdateRequest_UpdateType.REPLACE,
                  value: fromJson(ValueSchema, toJson(NodeDataSchema, node.state)),
                }),
              );
            }
          }
        }
      });

      localNodes.forEach((ln) => {
        if (!remoteNodeIds.has(ln.id)) {
          allMutations.push(createProto(RemoveNodeRequestSchema, { id: ln.id }));
        }
      });

      if (allMutations.length > 0) {
        applyMutations(allMutations, { source: MutationSource.SOURCE_SYNC });
      }
    } finally {
      processingRemoteUpdateRef.current = false;
    }
  }, [stNodes, stEdges, getConnection]);
};
