import { create } from "@bufbuild/protobuf";
import { type FileUIPart } from "ai";
import { useCallback } from "react";
import { toast } from "react-hot-toast";
import { v4 as uuidv4 } from "uuid";

import { ActionExecutionRequestSchema } from "../../../generated/flowcraft/v1/core/action_pb";
import { socketClient } from "../../../utils/SocketClient";
import { type ChatMessage, type ChatStatus, type ContextNode } from "./types";

export function useChatActions(
  nodeId: string,
  setStatus: (s: ChatStatus) => void,
  setHistory: React.Dispatch<React.SetStateAction<ChatMessage[]>>,
) {
  const uploadFile = async (file: File): Promise<null | string> => {
    const formData = new FormData();
    formData.append("file", file);
    try {
      const response = await fetch("/api/upload", {
        body: formData,
        method: "POST",
      });
      const asset = (await response.json()) as { url: string };
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
            new File([blob], file.filename ?? "img.png", {
              type: file.mediaType,
            }),
          );
          if (url) finalAttachments.push({ ...file, url });
        } else {
          finalAttachments.push(file);
        }
      }

      const userMsg: ChatMessage = {
        attachments: finalAttachments,
        content: content.trim(),
        contextNodes,
        createdAt: Date.now(),
        id: uuidv4(),
        role: "user",
      };

      setHistory((prev) => [...prev, userMsg]);

      try {
        await socketClient.send({
          payload: {
            case: "actionExecute",
            value: create(ActionExecutionRequestSchema, {
              actionId: "flowcraft.action.node.chat.generate",
              contextNodeIds: contextNodes.map((n) => n.id),
              params: {
                case: "chatGenerate",
                value: {
                  endpointId: "", // Optional
                  modelId: selectedModel,
                  userContent: content.trim(),
                  useWebSearch: useWebSearch,
                },
              },
              sourceNodeId: nodeId,
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
