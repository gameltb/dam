import { create as createProto } from "@bufbuild/protobuf";
import { useReactFlow } from "@xyflow/react";
import { useCallback, useEffect, useRef } from "react";
import { useSpacetimeDB, useTable } from "spacetimedb/react";

import { GraphMutationSchema } from "@/generated/flowcraft/v1/core/service_pb";
import { wrapReducers } from "@/utils/pb-client";
import { DbConnection, tables } from "@/generated/spacetime";
import { useFlowStore } from "@/store/flowStore";
import { useTaskStore } from "@/store/taskStore";
import { MutationSource, TaskStatus, TaskType } from "@/types";
import { convertStdbToPb } from "@/utils/pb-client";

const mapSpacetimeStatusToTaskStatus = (stStatus: { tag: string }): TaskStatus => {
  switch (stStatus.tag) {
    case "TASK_CANCELLED":
      return TaskStatus.TASK_CANCELLED;
    case "TASK_COMPLETED":
      return TaskStatus.TASK_COMPLETED;
    case "TASK_FAILED":
      return TaskStatus.TASK_FAILED;
    case "TASK_PENDING":
      return TaskStatus.TASK_PENDING;
    case "TASK_PROCESSING":
      return TaskStatus.TASK_PROCESSING;
    default:
      return TaskStatus.TASK_PENDING;
  }
};

export const useSpacetimeSync = () => {
  const stdb = useSpacetimeDB();
  const { isActive } = stdb;

  const getConnection = useCallback(() => stdb.getConnection<DbConnection>(), [stdb]);

  const [stNodes] = useTable(tables.nodes);
  const [stEdges] = useTable(tables.edges);
  const [stTasks] = useTable(tables.tasks);

  const { setViewport } = useReactFlow();
  const updateTask = useTaskStore((state) => state.updateTask);
  const registerTask = useTaskStore((state) => state.registerTask);

  const viewportInitializedRef = useRef(false);

  useEffect(() => {
    const conn = getConnection();
    if (conn && isActive) {
      const pbConn = wrapReducers(conn);
      useFlowStore.setState({ spacetimeConn: pbConn });

      const sessionTaskId = `user-session-${crypto.randomUUID()}`;
      pbConn.reducers.assignCurrentTask({ taskId: sessionTaskId });

      void conn
        .subscriptionBuilder()
        .subscribe([
          "SELECT * FROM nodes",
          "SELECT * FROM edges",
          "SELECT * FROM viewport_state",
          "SELECT * FROM tasks",
          "SELECT * FROM chat_messages",
        ]);

      const onViewportInsert = (_ctx: unknown, row: { id: string; state: unknown }) => {
        if (row.id === "default" && !viewportInitializedRef.current) {
          const viewportState = convertStdbToPb("viewportState", row);
          if (viewportState) {
            void setViewport({
              x: viewportState.x,
              y: viewportState.y,
              zoom: viewportState.zoom,
            });
            viewportInitializedRef.current = true;
          }
        }
      };

      conn.db.viewportState.onInsert(onViewportInsert);
    }
  }, [getConnection, isActive, setViewport]);

  useEffect(() => {
    stTasks.forEach((stTask) => {
      const taskStore = useTaskStore.getState();
      const existingTask = taskStore.tasks[stTask.id];
      const status = stTask.status as { tag: string };
      const newStatus = mapSpacetimeStatusToTaskStatus(status);

      if (!existingTask) {
        registerTask({
          label: `Remote Action: ${stTask.request.actionId}`,
          nodeId: stTask.request.sourceNodeId,
          status: newStatus,
          taskId: stTask.id,
          type: TaskType.REMOTE,
        });
      } else if (existingTask.status !== newStatus) {
        updateTask(stTask.id, {
          message: typeof stTask.result === "string" ? stTask.result : JSON.stringify(stTask.result),
          status: newStatus,
        });
      }
    });
  }, [stTasks, updateTask, registerTask]);

  useEffect(() => {
    if (stNodes.length === 0 && stEdges.length === 0) return;

    const { applyMutations, nodes: localNodes } = useFlowStore.getState();
    const remoteNodeIds = new Set(stNodes.map((n) => n.nodeId));
    const conn = getConnection();

    stNodes.forEach((nodeRow) => {
      if (!conn) return;
      const node = convertStdbToPb("nodes", nodeRow);
      if (!node) return;

      const localNode = localNodes.find((ln) => ln.id === node.nodeId);

      const state = node.state;
      const presentation = node.presentation;

      if (!localNode) {
        applyMutations(
          [
            createProto(GraphMutationSchema, {
              operation: {
                case: "addNode",
                value: {
                  node: {
                    isSelected: node.isSelected,
                    nodeId: node.nodeId,
                    nodeKind: node.nodeKind,
                    presentation: presentation,
                    state: state,
                    templateId: node.templateId,
                  },
                },
              },
            }),
          ],
          { source: MutationSource.SOURCE_SYNC },
        );
      } else {
        const isInteracting = (localNode.dragging ?? false) || (localNode.resizing ?? false);
        if (!isInteracting) {
          applyMutations(
            [
              createProto(GraphMutationSchema, {
                operation: {
                  case: "updateNode",
                  value: {
                    data: state,
                    id: node.nodeId,
                    presentation: presentation,
                  },
                },
              }),
            ],
            { source: MutationSource.SOURCE_SYNC },
          );
        }
      }
    });

    localNodes.forEach((ln) => {
      if (!remoteNodeIds.has(ln.id)) {
        applyMutations(
          [
            createProto(GraphMutationSchema, {
              operation: {
                case: "removeNode",
                value: { id: ln.id },
              },
            }),
          ],
          { source: MutationSource.SOURCE_SYNC },
        );
      }
    });
  }, [stNodes, stEdges]);
};
