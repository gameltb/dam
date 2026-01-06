import { useState, useEffect, useRef } from "react";
import { type ChatMessage } from "./types";
import { socketClient } from "../../../utils/SocketClient";

export function useChatHistory(conversationHeadId: string | undefined) {
  const [history, setHistory] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  // Keep track of history to check against optimistic updates
  const historyRef = useRef(history);
  useEffect(() => {
    historyRef.current = history;
  }, [history]);

  useEffect(() => {
    if (!conversationHeadId) {
      // Only clear if we actually have history (to avoid unnecessary renders)
      // and if we are not in the middle of an optimistic update that hasn't been persisted yet?
      // Actually, if headId is undefined, it means no conversation on backend.
      // But if we just started a conversation, headId is undefined until backend confirms.
      // So clearing here is dangerous if we have local unsaved messages.

      // However, usually headId becomes defined once backend replies.
      // If we switch nodes, headId changes.

      // Safety: If history is empty, it's fine.
      if (historyRef.current.length > 0) {
        // Check if we are potentially in a "new conversation" state locally
        // We can't easily know if the local history is "unsaved" or "old".
        // But typically if headId is undefined, it means "empty conversation".
        // We'll clear for now, but rely on the fact that optimistic updates usually happen
        // *before* headId updates, so headId won't go from "Something" to "Undefined"
        // unless we explicitly deleted everything.
        setHistory([]);
      }
      return;
    }

    // Check if our local history's last message matches the new headId.
    // If so, we assume our local state is up to date (optimistic update succeeded)
    // and we don't need to re-fetch, avoiding race conditions.
    const lastMsg = historyRef.current[historyRef.current.length - 1];
    if (lastMsg?.id === conversationHeadId) {
      // We are up to date.
      return;
    }

    const fetchHistory = async () => {
      setIsLoading(true);
      try {
        const res = await socketClient.getChatHistory(conversationHeadId);
        const mapped: ChatMessage[] = res.entries.map((m) => ({
          id: m.id,
          role: m.role as any,
          content: m.content,
          createdAt: Number(m.timestamp),
        }));
        setHistory(mapped);
      } catch (e) {
        console.error("[useChatHistory] Failed to fetch:", e);
      } finally {
        setIsLoading(false);
      }
    };

    void fetchHistory();
  }, [conversationHeadId]);

  return { history, setHistory, isLoading };
}
