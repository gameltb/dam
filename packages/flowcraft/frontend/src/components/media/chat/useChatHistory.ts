import { create } from "@bufbuild/protobuf";
import { useEffect, useRef, useState } from "react";

import { ChatMessagePartSchema } from "@/generated/flowcraft/v1/actions/chat_actions_pb";
import { socketClient } from "@/utils/SocketClient";
import { type ChatMessage } from "./types";
import { partsToText } from "./utils";

export function useChatHistory(
  conversationHeadId: string | undefined,
  optimisticContent?: string,
  isStreamDone?: boolean,
  shouldFetch = true,
) {
  const [history, setHistory] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  // Keep track of history to check against optimistic updates
  const historyRef = useRef(history);
  useEffect(() => {
    historyRef.current = history;
  }, [history]);

  useEffect(() => {
    if (!shouldFetch || !conversationHeadId) {
      if (!conversationHeadId && historyRef.current.length > 0) {
        setHistory([]);
      }
      return;
    }

    const currentHistory = historyRef.current;
    const lastMsg = currentHistory[currentHistory.length - 1];

    // 1. If the current head matches our last message ID, we are perfectly synced.
    if (lastMsg?.id === conversationHeadId) {
      return;
    }

    // 2. If we just finished a stream and the new conversationHeadId is different from our last message,
    // it's likely the ID of the assistant message we just streamed.
    if (isStreamDone && optimisticContent && lastMsg) {
      // Check if we already added this assistant message (to prevent double append)
      const lastMsgText = partsToText(lastMsg.parts);
      const alreadyHasAssistant =
        lastMsg.role === "assistant" && lastMsgText === optimisticContent;

      if (!alreadyHasAssistant) {
        const optimisticMsg: ChatMessage = {
          createdAt: Date.now(),
          id: conversationHeadId, // Use the real server ID provided by the mutation
          parentId: lastMsg.id,
          parts: [
            create(ChatMessagePartSchema, {
              part: { case: "text", value: optimisticContent },
            }),
          ],
          role: "assistant",
          siblingIds: [],
        };
        setHistory((prev) => [...prev, optimisticMsg]);
        return; // Successfully transitioned locally, SKIP FETCH
      } else if (lastMsg.id !== conversationHeadId) {
        // If we have the message but the ID was temporary/wrong, update it quietly
        setHistory((prev) =>
          prev.map((m, i) =>
            i === prev.length - 1 ? { ...m, id: conversationHeadId } : m,
          ),
        );
        return;
      }
    }

    // 3. Fallback: Fetch full history only if we can't reconcile locally
    const fetchHistory = async () => {
      setIsLoading(true);
      try {
        const res = await socketClient.getChatHistory(conversationHeadId);
        const mapped: ChatMessage[] = res.entries.map((m) => {
          let metadata: Record<string, unknown> = {};
          if (m.metadata.case === "chatMetadata") {
            metadata = {
              attachments: m.metadata.value.attachmentUrls,
              modelId: m.metadata.value.modelId,
            };
          } else if (m.metadata.case === "metadataStruct") {
            metadata = m.metadata.value as Record<string, unknown>;
          }

          return {
            createdAt: Number(m.timestamp),
            id: m.id,
            metadata,
            parentId: m.parentId || undefined,
            parts: m.parts,
            role: (["assistant", "system", "user"].includes(m.role)
              ? m.role
              : "user") as "assistant" | "system" | "user",
            siblingIds: m.siblingIds,
          };
        });
        setHistory(mapped);
      } catch (e) {
        console.error("[useChatHistory] Failed to fetch:", e);
      } finally {
        setIsLoading(false);
      }
    };

    void fetchHistory();
  }, [conversationHeadId, optimisticContent, isStreamDone, shouldFetch]);

  return { history, isLoading, setHistory };
}
