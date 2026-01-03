import { type GraphMiddleware } from "./types";
import { useTaskStore } from "../taskStore";
import { MutationSource } from "../../generated/flowcraft/v1/core/base_pb";
import { v4 as uuidv4 } from "uuid";

/**
 * TaskMiddleware
 * 负责记录变更到任务日志中，并确保关联的任务存在
 */
export const taskMiddleware: GraphMiddleware = (event, next) => {
  const { mutations, context } = event;
  const source = context.source ?? MutationSource.SOURCE_USER;
  const taskId = context.taskId ?? "manual-action";

  // 1. 确保任务在 taskStore 中注册
  if (!useTaskStore.getState().tasks[taskId]) {
    useTaskStore.getState().registerTask({
      taskId,
      label: context.taskId ? `Task ${taskId}` : "Manual Action",
      source: source,
    });
  }

  // 2. 添加变更日志
  const logId = uuidv4();
  useTaskStore.getState().addMutationLog({
    id: logId,
    taskId,
    source: source,
    timestamp: Date.now(),
    description: context.description ?? "Graph updated",
    mutations,
  });

  // 3. 将日志链接到任务
  useTaskStore.getState().linkMutationToTask(taskId, logId);

  next(event);
};
