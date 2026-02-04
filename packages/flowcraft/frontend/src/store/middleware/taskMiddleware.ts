import { v4 as uuidv4 } from "uuid";

import { MutationSource } from "@/generated/flowcraft/v1/core/base_pb";

import { useTaskStore } from "../taskStore";
import { type GraphMiddleware, type GraphMutationEvent } from "./types";

/**
 * TaskMiddleware
 * Responsible for recording changes to the task log (in JSON text format).
 */
export const taskMiddleware: GraphMiddleware = (event: GraphMutationEvent, next) => {
  const { context, patches } = event;
  const source = context.source ?? MutationSource.SOURCE_USER;
  const taskId = context.taskId ?? "manual-action";

  // 1. Ensure the task is registered in taskStore
  if (!useTaskStore.getState().tasks[taskId]) {
    useTaskStore.getState().registerTask({
      label: context.taskId ? `Task ${taskId}` : "Manual Action",
      source: source,
      taskId,
    });
  }

  // 2. Convert the Patch list to a JSON string
  const mutationsJson = JSON.stringify(patches);

  // 3. Add mutation log
  const logId = uuidv4();
  useTaskStore.getState().addMutationLog({
    description: context.description ?? "Graph updated",
    id: logId,
    mutationsJson,
    source: source,
    taskId,
    timestamp: Date.now(),
  });

  // 4. Link the log to the task
  useTaskStore.getState().linkMutationToTask(taskId, logId);

  next(event);
};
