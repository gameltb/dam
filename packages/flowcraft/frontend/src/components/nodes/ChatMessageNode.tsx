import { Handle, type NodeProps, Position } from "@xyflow/react";
import { memo } from "react";
import ReactMarkdown from "react-markdown";

import { partsToText } from "@/components/media/chat/utils";
import { cn } from "@/lib/utils";

export const ChatMessageNode = memo(({ data, selected }: NodeProps<any>) => {
  const { metadata } = data;
  const role = metadata?.role || "unknown";
  const partsJson = metadata?.parts_json;
  const createdAtStr = metadata?.timestamp || Date.now().toString();

  const isUser = role === "user";
  let content = "No content";

  try {
    if (partsJson) {
      const parts = JSON.parse(partsJson);
      content = partsToText(parts);
    }
  } catch (e) {
    console.error("Failed to parse message parts", e);
  }

  return (
    <div
      className={cn(
        "px-4 py-3 rounded-lg border-2 min-w-[200px] max-w-[300px] shadow-sm transition-all bg-background",
        selected ? "border-primary ring-2 ring-primary/20" : "border-node-border",
        isUser ? "bg-primary/5" : "bg-muted/30",
      )}
    >
      <Handle className="!w-2 !h-2 !bg-primary-color" position={Position.Top} type="target" />

      <div className="flex flex-col gap-1">
        <div className="flex justify-between items-center mb-1">
          <span
            className={cn(
              "text-[10px] font-bold uppercase px-1.5 py-0.5 rounded",
              isUser ? "bg-primary/20 text-primary" : "bg-muted text-muted-foreground",
            )}
          >
            {role}
          </span>
          <span className="text-[9px] opacity-40">{new Date(Number(createdAtStr)).toLocaleTimeString()}</span>
        </div>

        <div className="text-xs prose prose-invert max-h-[200px] overflow-y-auto scrollbar-none break-words">
          <ReactMarkdown>{content}</ReactMarkdown>
        </div>
      </div>

      <Handle className="!w-2 !h-2 !bg-primary-color" position={Position.Bottom} type="source" />
    </div>
  );
});

ChatMessageNode.displayName = "ChatMessageNode";
