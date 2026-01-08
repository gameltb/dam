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
  const emitMutation = (m: MutationList) => {
    console.log(
      `[NodeExecutor] Node ${nodeId} emitted mutation${actionId ? ` via action ${actionId}` : ""}`,
    );
    m.mutations.forEach((mut) => {
      executeMutation(mut, serverGraph);
    });
    incrementVersion();
    eventBus.emit("mutations", m);
  };

  const emitNodeEvent = (e: NodeEvent) => {
    console.log(
      `[NodeExecutor] Node ${nodeId} emitted event: ${e.payload.case ?? "unknown"}`,
    );
    eventBus.emit("nodeEvent", e);
  };

  const emitWidgetStream = (
    widgetId: string,
    chunk: string,
    isDone = false,
  ) => {
    emitNodeEvent(
      create(NodeEventSchema, {
        nodeId: nodeId,
        payload: {
          case: "widgetStream",
          value: {
            chunkData: chunk,
            isDone,
            widgetId,
          },
        },
      }),
    );
  };

  const emitTaskUpdate = (t: TaskUpdate) => {
    console.log(
      `[NodeExecutor] Node ${nodeId} emitted task update: ${t.taskId}`,
    );
    // Ensure nodeId is attached for frontend filtering
    t.nodeId = nodeId;
    eventBus.emit("taskUpdate", t);
  };

  return { emitMutation, emitNodeEvent, emitTaskUpdate, emitWidgetStream };
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
  const node = serverGraph.nodes.find((n) => n.id === nodeId);
  if (!node || !isDynamicNode(node)) return;

  const typeId = node.data.typeId ?? "";

  // Handle explicit restart signal
  if (payload.case === "restartInstance") {
    console.log(`[NodeExecutor] Force restarting instances for node ${nodeId}`);
    await instanceHost.stopAllForNode(nodeId);
    // FALLTHROUGH: The code below will recreate the instance if needed
  }

  // 1. Try to find existing persistent instance
  let instance = instanceHost
    .getInstancesForNode(nodeId)
    .find((i) => i instanceof NodeInstance);

  // 2. If not exists, try to create one if defined in registry
  if (!instance) {
    const def = NodeRegistry.getDefinition(typeId);
    if (def?.createInstance) {
      instance = def.createInstance(nodeId);
      instanceHost.registerInstance(instance);
      await instance.start({});
    }
  }

  // 3. Dispatch signal to instance or fallback to stateless executor
  if (instance) {
    await instance.handleSignal(payload);
  } else {
    const executor = NodeRegistry.getExecutor(typeId);
    if (executor) {
      const taskId = uuidv4();
      const { emitMutation, emitNodeEvent, emitTaskUpdate, emitWidgetStream } =
        createNodeEmitter(nodeId);
      await executor({
        emitMutation,
        emitNodeEvent,
        emitTaskUpdate,
        emitWidgetStream,
        node,
        params: payload,
        taskId,
      });
    }
  }
}
