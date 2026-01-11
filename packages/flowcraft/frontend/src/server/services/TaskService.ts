import { create } from "@bufbuild/protobuf";

import {
  ChatActionParamsSchema,
  ChatEditParamsSchema,
} from "@/generated/flowcraft/v1/actions/chat_actions_pb";
import { ActionId, NodeSignalCase } from "@/types";

import { runNodeSignal } from "./NodeExecutor";
import { getSpacetimeConn } from "../spacetimeClient";
import logger from "../utils/logger";

const ACTION_HANDLERS: {
  [K in ActionId]?: (task: any) => Promise<void>;
} = {
  [ActionId.CHAT_GENERATE]: async (task) => {
    const params = JSON.parse(task.paramsJson);
    await runNodeSignal(task.nodeId, {
      case: NodeSignalCase.CHAT_GENERATE,
      value: create(ChatActionParamsSchema, {
        endpointId: params.endpointId || "openai",
        modelId: params.modelId || "gpt-4o",
        userContent: params.userContent || "",
      }),
    });
  },
  [ActionId.CHAT_EDIT]: async (task) => {
    const params = JSON.parse(task.paramsJson);
    await runNodeSignal(task.nodeId, {
      case: NodeSignalCase.CHAT_EDIT,
      value: create(ChatEditParamsSchema, {
        messageId: params.messageId,
        newParts: params.newParts,
      }),
    });
  },
  [ActionId.CHAT_DUPLICATE_BRANCH]: async (task) => {
    const params = JSON.parse(task.paramsJson);
    const { duplicateBranch } = await import("./ChatService");
    duplicateBranch({
      newParentId: params.newParentId || null,
      newTreeId: params.treeId,
      nodeId: task.nodeId,
      startMessageId: params.sourceHeadId,
    });
  },
};

import { createTaskConnection } from "../spacetimeClient";

export async function handleTask(task: any) {
  // 1. Create a dedicated connection for this task to establish context
  const taskConn = await createTaskConnection(task.id);
  
  logger.info(`Processing task: ${task.id} (Action: ${task.actionId}) using isolated client.`);

  try {
    const actionId = task.actionId as ActionId;
    const handler = ACTION_HANDLERS[actionId];

    if (handler) {
      await handler(task);
    } else {
      logger.warn(`[TaskService] No handler defined for action: ${actionId}`);
    }

    taskConn.reducers.updateTaskStatus({
      id: task.id,
      resultJson: JSON.stringify({ success: true }),
      status: "completed",
    });
  } catch (err: any) {
    logger.error(`Task ${task.id} failed:`, err);
    taskConn.reducers.updateTaskStatus({
      id: task.id,
      resultJson: String(err),
      status: "failed",
    });
  } finally {
    // 2. Disconnect the task client. onDisconnect in STDB module will clean up assignment table.
    taskConn.disconnect();
  }
}

export function initTaskWatcher() {
  const conn = getSpacetimeConn();
  if (!conn) return;

  conn.db.tasks.onInsert((_ctx: any, row: any) => {
    if (row.status === "pending") {
      void handleTask(row);
    }
  });

  conn.db.tasks.onUpdate((_ctx: any, oldRow: any, newRow: any) => {
    if (oldRow.status !== "pending" && newRow.status === "pending") {
      void handleTask(newRow);
    }
  });
}