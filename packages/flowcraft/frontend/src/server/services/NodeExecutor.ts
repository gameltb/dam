import { create } from "@bufbuild/protobuf";
import { v4 as uuidv4 } from "uuid";

import { type ActionExecutionRequest } from "@/generated/flowcraft/v1/core/action_pb";
import { type TaskUpdate } from "@/generated/flowcraft/v1/core/node_pb";
import {
  type MutationList,
  type NodeEvent,
  NodeEventSchema,
} from "@/generated/flowcraft/v1/core/service_pb";
import { type NodeSignal } from "@/generated/flowcraft/v1/core/signals_pb";
import { isDynamicNode } from "@/types";

import { getSpacetimeConn } from "../spacetimeClient";
import logger from "../utils/logger";
import { ActionInstance } from "./ActionInstance";
import { instanceHost } from "./InstanceHost";
import { executeMutation } from "./MutationExecutor";
import { NodeInstance } from "./NodeInstance";
import { NodeRegistry } from "./NodeRegistry";
import { eventBus, incrementVersion, serverGraph } from "./PersistenceService";

export interface ExecutionContext {
  actionId?: string;
  nodeId: string;
  params: unknown;
  sourceNodeId?: string;
}

export function createNodeEmitter(nodeId: string, actionId?: string) {
  const conn = getSpacetimeConn();

  const emitMutation = (m: MutationList) => {
    logger.info(
      `Node ${nodeId} emitted mutation${actionId ? ` via action ${actionId}` : ""}`,
    );

    m.mutations.forEach((mut) => {
      executeMutation(mut, serverGraph);
      // STDB sync happens automatically via Reducer calls if implemented here
    });

    incrementVersion();
    eventBus.emit("mutations", m);
  };

  const emitNodeEvent = (e: NodeEvent) => {
    logger.info(`Node ${nodeId} emitted event: ${e.payload.case ?? "unknown"}`);
    eventBus.emit("nodeEvent", e);
  };

  const emitWidgetStream = (widgetId: string, chunk: string, isDone = false) => {
    if (conn && isDone) {
      conn.reducers.updateWidgetValue({
        nodeId: nodeId,
        valueJson: JSON.stringify(chunk),
        widgetId: widgetId,
      });
    }

    emitNodeEvent(
      create(NodeEventSchema, {
        nodeId: nodeId,
        payload: {
          case: "widgetStream",
          value: { chunkData: chunk, isDone, widgetId },
        },
      }),
    );
  };

  const emitTaskUpdate = (t: TaskUpdate) => {
    logger.info(`Node ${nodeId} emitted task update: ${t.taskId}`);
    if (conn) {
      conn.reducers.updateTaskStatus({
        id: t.taskId,
        resultJson: t.result ? JSON.stringify(t.result) : "",
        status: mapProtoTaskStatus(t.status),
      });
    }
    t.nodeId = nodeId;
    eventBus.emit("taskUpdate", t);
  };

  const emitChatStream = (
    content: string,
    status: "idle" | "streaming" | "thinking",
    parentId = "",
  ) => {
    if (conn) {
      conn.reducers.updateChatStream({ content, nodeId, parentId, status });
    }
  };

  return {
    emitChatStream,
    emitMutation,
    emitNodeEvent,
    emitTaskUpdate,
    emitWidgetStream,
  };
}

export async function runAction(
  actionId: string,
  sourceNodeId: string,
  params: ActionExecutionRequest["params"],
  contextNodeIds: string[] = [],
) {
  const handler = NodeRegistry.getActionHandler(actionId);
  const taskId = uuidv4();
  const node = serverGraph.nodes.find((n) => n.id === sourceNodeId);

  if (handler) {
    const action = new ActionInstance(
      taskId,
      actionId,
      async (actionCtx) => {
        const { emitMutation, emitNodeEvent, emitTaskUpdate } =
          createNodeEmitter(sourceNodeId, actionId);
        await handler({
          actionId,
          contextNodeIds,
          emitMutation,
          emitNodeEvent,
          emitTaskUpdate,
          node,
          params,
          selectedNodeIds: [],
          sourceNodeId,
          taskId: actionCtx.taskId,
        });
      },
      sourceNodeId,
    );
    instanceHost.registerInstance(action);
    await action.start({});
  } else if (node && isDynamicNode(node)) {
    const executor = NodeRegistry.getExecutor(node.data.typeId ?? "");
    if (executor) {
      const action = new ActionInstance(
        taskId,
        `${actionId} on ${node.id}`,
        async (actionCtx) => {
          const emitters = createNodeEmitter(sourceNodeId, actionId);
          await executor({
            actionId,
            ...emitters,
            node,
            params,
            taskId: actionCtx.taskId,
          });
        },
        sourceNodeId,
      );
      instanceHost.registerInstance(action);
      await action.start({});
    }
  }
}

export async function runNodeSignal(
  nodeId: string,
  payload: NodeSignal["payload"],
) {
  const conn = getSpacetimeConn();
  if (!conn) return;

  const stNode = conn.db.nodes.id.find(nodeId);
  if (!stNode) return;

  let typeId = stNode.templateId;
  try {
    const data = JSON.parse(stNode.dataJson);
    if (data.typeId) typeId = data.typeId;
  } catch {}

  if (payload.case === "restartInstance") {
    await instanceHost.stopAllForNode(nodeId);
  }

  let instance = instanceHost
    .getInstancesForNode(nodeId)
    .find((i) => i instanceof NodeInstance);

  if (!instance) {
    const def = NodeRegistry.getDefinition(typeId);
    if (def?.createInstance) {
      instance = def.createInstance(nodeId);
      instanceHost.registerInstance(instance);
      await instance.start({});
    }
  }

  if (instance) {
    await instance.handleSignal(payload);
  } else {
    const executor = NodeRegistry.getExecutor(typeId);
    if (executor) {
      const taskId = uuidv4();
      const emitters = createNodeEmitter(nodeId);
      await executor({
        ...emitters,
        node: { data: {}, id: nodeId, type: "dynamic" } as any,
        params: payload,
        taskId,
      });
    }
  }
}

function mapProtoTaskStatus(status: number): string {
  switch (status) {
    case 0:
      return "pending";
    case 1:
      return "processing";
    case 2:
      return "completed";
    case 3:
      return "failed";
    case 4:
      return "cancelled";
    default:
      return "unknown";
  }
}