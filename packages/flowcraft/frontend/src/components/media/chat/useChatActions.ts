import { useCallback } from "react";
import { v4 as uuidv4 } from "uuid";
import { toast } from "react-hot-toast";
import { create } from "@bufbuild/protobuf";
import { type FileUIPart } from "ai";
import { type ChatMessage, type ContextNode, type ChatStatus } from "./types";
import { socketClient } from "../../../utils/SocketClient";
import { ActionExecutionRequestSchema } from "../../../generated/flowcraft/v1/core/action_pb";

export function useChatActions(
  nodeId: string,
  setStatus: (s: ChatStatus) => void,
  setHistory: React.Dispatch<React.SetStateAction<ChatMessage[]>>,
) {
  const uploadFile = async (file: File): Promise<string | null> => {
    const formData = new FormData();
    formData.append("file", file);
    try {
      const response = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });
      const asset = await response.json();
      return asset.url;
    } catch (err) {
      console.error("Upload failed:", err);
      return null;
    }
  };

  const sendMessage = useCallback(
    async (
      content: string,
      selectedModel: string,
      useWebSearch: boolean,
      files: FileUIPart[] = [],
      contextNodes: ContextNode[] = [],
    ) => {
      setStatus("submitted");

      const finalAttachments: FileUIPart[] = [];
      for (const file of files) {
        if (file.url.startsWith("blob:")) {
          const response = await fetch(file.url);
          const blob = await response.blob();
          const url = await uploadFile(
            new File([blob], file.filename || "img.png", {
              type: file.mediaType,
            }),
          );
          if (url) finalAttachments.push({ ...file, url });
        } else {
          finalAttachments.push(file);
        }
      }

      const userMsg: ChatMessage = {
        id: uuidv4(),
        role: "user",
        content: content.trim(),
        createdAt: Date.now(),
        attachments: finalAttachments,
        contextNodes,
      };

      setHistory((prev) => [...prev, userMsg]);

      try {
        await socketClient.send({
          payload: {
            case: "actionExecute",
            value: create(ActionExecutionRequestSchema, {
              actionId: "chat:generate",
              sourceNodeId: nodeId,
              contextNodeIds: contextNodes.map((n) => n.id),
              paramsJson: JSON.stringify({
                model: selectedModel,
                webSearch: useWebSearch,
                messages: [userMsg],
                stream: true,
              }),
            }),
          },
        });
      } catch (err) {
        console.error("Send failed:", err);
        setStatus("ready");
        toast.error("Failed to send message");
      }
    },
    [nodeId, setStatus, setHistory],
  );

  return { sendMessage };
}
