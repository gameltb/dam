import { useEffect, useRef } from "react";
import { useTable } from "spacetimedb/react";

import { type ChatMessage } from "@/generated/flowcraft/v1/core/service_pb";
import { tables } from "@/generated/spacetime";
import { ChatStreamStatus, useChatStore } from "@/store/chatStore";
import { useFlowStore } from "@/store/flowStore";
import { useTaskStore } from "@/store/taskStore";
import { TaskStatus, TaskType } from "@/types";
import { convertStdbToPb } from "@/utils/pb-client";

export const useTaskSync = (isActive: boolean) => {
  const spacetimeConn = useFlowStore((s) => s.spacetimeConn);
  const [stTasks] = useTable(tables.tasks);
  const [stChatMessages] = useTable(tables.chatMessages);
  const [stChatStreams] = useTable(tables.chatStreams);

  const updateTask = useTaskStore((state) => state.updateTask);
  const registerTask = useTaskStore((state) => state.registerTask);
  const setChatMessages = useChatStore((state) => state.setMessages);
  const setChatStreams = useChatStore((state) => state.setStreams);

  // Chat Messages Sync
  const lastProcessedMessagesRef = useRef<string>("");
  useEffect(() => {
    if (!isActive) return;
    const messagesJson = JSON.stringify(stChatMessages);
    if (messagesJson === lastProcessedMessagesRef.current) return;
    lastProcessedMessagesRef.current = messagesJson;

    const pbMessages = stChatMessages
      .map((row) => {
        const pb = convertStdbToPb(
          "chatMessages",
          row as Record<string, unknown>,
          spacetimeConn?.db,
        ) as ChatMessage | null;
        if (pb && !pb.id && (row as any).id) {
          pb.id = (row as any).id as string;
        }
        return pb;
      })
      .filter((m): m is ChatMessage => m !== null);
    setChatMessages(pbMessages);
  }, [stChatMessages, setChatMessages, isActive, spacetimeConn]);

  // Chat Streams Sync
  const lastProcessedStreamsRef = useRef<string>("");
  useEffect(() => {
    if (!isActive) return;
    const streamsJson = JSON.stringify(stChatStreams);
    if (streamsJson === lastProcessedStreamsRef.current) return;
    lastProcessedStreamsRef.current = streamsJson;

    const streams = stChatStreams.map((row) => ({
      content: row.content,
      nodeId: row.nodeId,
      status: row.status as ChatStreamStatus,
    }));
    setChatStreams(streams);
  }, [stChatStreams, setChatStreams, isActive]);

  // Tasks Sync
  const lastProcessedTasksRef = useRef<string>("");
  useEffect(() => {
    if (!isActive) return;
    const tasksJson = JSON.stringify(stTasks);
    if (tasksJson === lastProcessedTasksRef.current) return;
    lastProcessedTasksRef.current = tasksJson;

    stTasks.forEach((stTask) => {
      const taskStore = useTaskStore.getState();
      const existingTask = taskStore.tasks[stTask.id];
      const newStatus = stTask.status as unknown as TaskStatus;
      const currentResult = stTask.result || (newStatus === TaskStatus.PENDING ? "Initializing…" : "");

      if (!existingTask) {
        registerTask({
          label: `Task: ${stTask.taskType}`,
          message: currentResult,
          nodeId: stTask.nodeId,
          status: newStatus,
          taskId: stTask.id,
          type: TaskType.REMOTE,
        });
      } else if (existingTask.status !== newStatus || existingTask.message !== currentResult) {
        updateTask(stTask.id, {
          message: currentResult,
          status: newStatus,
        });
      }
    });
  }, [stTasks, updateTask, registerTask, isActive]);
};
