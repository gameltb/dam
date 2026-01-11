import { create as createProto } from "@bufbuild/protobuf";
import { useReactFlow } from "@xyflow/react";
import { useEffect, useRef } from "react";
import { useSpacetimeDB, useTable } from "spacetimedb/react";

import { PresentationSchema } from "@/generated/flowcraft/v1/core/base_pb";
import { GraphMutationSchema } from "@/generated/flowcraft/v1/core/service_pb";
import { DbConnection, tables } from "@/generated/spacetime";
import { useFlowStore } from "@/store/flowStore";
import { useTaskStore } from "@/store/taskStore";
import {
  type DynamicNodeData,
  MutationSource,
  TaskStatus,
  TaskType,
} from "@/types";
import { toProtoNodeData } from "@/utils/protoAdapter";

const mapSpacetimeStatusToTaskStatus = (stStatus: string): TaskStatus => {
  switch (stStatus) {
    case "cancelled":
      return TaskStatus.TASK_CANCELLED;
    case "completed":
      return TaskStatus.TASK_COMPLETED;
    case "failed":
      return TaskStatus.TASK_FAILED;
    case "pending":
      return TaskStatus.TASK_PENDING;
    case "processing":
      return TaskStatus.TASK_PROCESSING;
    default:
      return TaskStatus.TASK_PENDING;
  }
};

export const useSpacetimeSync = () => {
  const { connectionError, getConnection, identity, isActive } =
    useSpacetimeDB();
  const [stNodes] = useTable(tables.nodes);
  const [stEdges] = useTable(tables.edges);
  const [stWidgetValues] = useTable(tables.widgetValues);
  const [stTasks] = useTable(tables.tasks);

  const { setViewport } = useReactFlow();
  const applyMutations = useFlowStore((state) => state.applyMutations);
  const updateTask = useTaskStore((state) => state.updateTask);
  const registerTask = useTaskStore((state) => state.registerTask);

  const isSyncingRef = useRef(false);
  const viewportInitializedRef = useRef(false);

  useEffect(() => {
    if (isActive && identity) {
      console.log(
        "[SpacetimeSync] Connection established. Identity:",
        identity.toHexString(),
      );
    } else if (connectionError) {
      console.error("[SpacetimeSync] Connection error:", connectionError);
    }
  }, [isActive, identity, connectionError]);

  useEffect(() => {
    const conn = getConnection<DbConnection>();
    if (conn && isActive) {
      useFlowStore.setState({ spacetimeConn: conn });

      // Register session-wide User Task
      const sessionTaskId = `user-session-${crypto.randomUUID()}`;
      conn.reducers.assignCurrentTask({ taskId: sessionTaskId });

      conn
        .subscriptionBuilder()
        .subscribe([
          "SELECT * FROM nodes",
          "SELECT * FROM edges",
          "SELECT * FROM viewport_state",
          "SELECT * FROM widget_values",
          "SELECT * FROM tasks",
          "SELECT * FROM chat_messages",
          "SELECT * FROM chat_streams",
        ]);

      const onViewportInsert = (_ctx: any, row: any) => {
        if (row.id === "default" && !viewportInitializedRef.current) {
          setViewport({ x: row.x, y: row.y, zoom: row.zoom });
          viewportInitializedRef.current = true;
        }
      };

      conn.db.viewportState.onInsert(onViewportInsert);

      const existing = conn.db.viewportState.id.find("default");
      if (existing && !viewportInitializedRef.current) {
        setViewport({ x: existing.x, y: existing.y, zoom: existing.zoom });
        viewportInitializedRef.current = true;
      }
    }
  }, [getConnection, isActive, setViewport]);

  // Sync Tasks from Spacetime to TaskStore
  useEffect(() => {
    stTasks.forEach((stTask) => {
      const taskStore = useTaskStore.getState();
      const existingTask = taskStore.tasks[stTask.id];
      const newStatus = mapSpacetimeStatusToTaskStatus(stTask.status);

      if (!existingTask) {
        registerTask({
          label: `Remote Action: ${stTask.actionId}`,
          nodeId: stTask.nodeId,
          status: newStatus,
          taskId: stTask.id,
          type: TaskType.REMOTE,
        });
      } else if (existingTask.status !== newStatus) {
        updateTask(stTask.id, {
          message: stTask.status === "failed" ? stTask.resultJson : undefined,
          status: newStatus,
        });
      }
    });
  }, [stTasks, updateTask, registerTask]);

  // Differential Sync via applyMutations
  useEffect(() => {
    if (stNodes.length === 0 && stEdges.length === 0) return;
    if (isSyncingRef.current) return;

    const { edges: localEdges, nodes: localNodes } = useFlowStore.getState();
    const mutations: any[] = [];
    const remoteNodeIds = new Set(stNodes.map((n) => n.id));
    const remoteEdgeIds = new Set(stEdges.map((e) => e.id));

    const EPSILON = 0.5;

    // 1. Sync Nodes (Remote -> Local)
    stNodes.forEach((n: any) => {
      const localNode = localNodes.find((ln) => ln.id === n.id);
      const data = JSON.parse(n.dataJson);

      // Merge fine-grained widget values into data
      const relevantWidgets = stWidgetValues.filter((wv) => wv.nodeId === n.id);
      if (relevantWidgets.length > 0) {
        data.widgetsValues = data.widgetsValues || {};
        relevantWidgets.forEach((wv) => {
          data.widgetsValues[wv.widgetId] = JSON.parse(wv.valueJson);
        });
      }

      const presentation = createProto(PresentationSchema, {
        height: n.height,
        parentId: n.parentId || "",
        position: { x: n.posX, y: n.posY },
        width: n.width,
      });

      if (!localNode) {
        mutations.push(
          createProto(GraphMutationSchema, {
            operation: {
              case: "addNode",
              value: {
                node: {
                  isSelected: n.isSelected,
                  nodeId: n.id,
                  nodeKind: n.kind,
                  presentation,
                  state: toProtoNodeData(data as DynamicNodeData),
                  templateId: n.templateId,
                },
              },
            },
          }),
        );
      } else {
        const isInteracting =
          (localNode as any).dragging || (localNode as any).resizing;

        const posChanged =
          Math.abs(localNode.position.x - n.posX) > EPSILON ||
          Math.abs(localNode.position.y - n.posY) > EPSILON;

        const dataChanged =
          JSON.stringify(localNode.data) !== JSON.stringify(data);
        const layoutChanged =
          Math.abs((localNode.measured?.width ?? 0) - n.width) > EPSILON ||
          Math.abs((localNode.measured?.height ?? 0) - n.height) > EPSILON;

        if (!isInteracting && (posChanged || dataChanged || layoutChanged)) {
          mutations.push(
            createProto(GraphMutationSchema, {
              operation: {
                case: "updateNode",
                value: {
                  data: toProtoNodeData(data as DynamicNodeData),
                  id: n.id,
                  presentation,
                },
              },
            }),
          );
        }
      }
    });

    // 2. Remove Nodes
    localNodes.forEach((ln) => {
      if (!remoteNodeIds.has(ln.id)) {
        mutations.push(
          createProto(GraphMutationSchema, {
            operation: {
              case: "removeNode",
              value: { id: ln.id },
            },
          }),
        );
      }
    });

    // 3. Sync Edges
    stEdges.forEach((e: any) => {
      const localEdge = localEdges.find((le) => le.id === e.id);
      if (!localEdge) {
        mutations.push(
          createProto(GraphMutationSchema, {
            operation: {
              case: "addEdge",
              value: {
                edge: {
                  edgeId: e.id,
                  sourceHandle: e.sourceHandle,
                  sourceNodeId: e.sourceId,
                  targetHandle: e.targetHandle,
                  targetNodeId: e.targetId,
                },
              },
            },
          }),
        );
      }
    });

    // 4. Remove Edges
    localEdges.forEach((le) => {
      if (!remoteEdgeIds.has(le.id)) {
        mutations.push(
          createProto(GraphMutationSchema, {
            operation: {
              case: "removeEdge",
              value: { id: le.id },
            },
          }),
        );
      }
    });

    if (mutations.length > 0) {
      isSyncingRef.current = true;
      try {
        applyMutations(mutations, { source: MutationSource.SOURCE_SYNC });
      } finally {
        setTimeout(() => {
          isSyncingRef.current = false;
        }, 0);
      }
    }
  }, [stNodes, stEdges, applyMutations]);
};
