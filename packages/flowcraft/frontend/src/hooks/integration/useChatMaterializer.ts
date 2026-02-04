import { useEffect } from "react";

/**
 * useChatMaterializer
 * (Simplified for new architecture)
 */
export const initChatMaterializer = () => {
  console.log("[ChatMaterializer] Initialized");
};

export const useChatMaterializer = () => {
  useEffect(() => {
    // Subscription logic is now handled by useGraphSync and specific materializers
  }, []);
};