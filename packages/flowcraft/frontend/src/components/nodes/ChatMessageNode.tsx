import { Handle, type NodeProps, NodeResizer, Position } from "@xyflow/react";
import { memo, useCallback, useMemo } from "react";

import { useSyncedBinding } from "@/hooks/core/useSyncedBinding";
import { useUiProperty } from "@/hooks/core/useUiProperty";
import { useNodeMutation } from "@/hooks/nodes/useNodeMutation";
import { cn } from "@/lib/utils";
import { useUiStore } from "@/store/uiStore";
import { NodeLenses } from "@/utils/lenses";

import { NodeShell } from "../base/NodeShell";
import { MarkdownRenderer } from "../media/MarkdownRenderer";
import { NodeJumpOverlay } from "./parts/NodeJumpOverlay";

export const ChatMessageNode = memo(({ data, id, selected }: NodeProps) => {
  const typedData = data as { metadata?: { role?: string; timestamp?: string }; presentation?: { height?: number } };
  const metadata = typedData.metadata;
  const role = metadata?.role ?? "unknown";
  const createdAtStr = metadata?.timestamp ?? "";

  const [isEditing, setIsEditing] = useUiProperty(id, "isEditing", false);
  const { updateLayout } = useNodeMutation(id);

  // Bidirectional binding: message content
  const [content, setContent] = useSyncedBinding(useMemo(() => NodeLenses.messageContent(id), [id]));

  const isUser = role === "user";
  const setNavigatingNode = useUiStore((s) => s.setNavigatingNode);
  const resetNavigatingNode = useUiStore((s) => s.resetNavigatingNode);

  const handleResizeEnd = useCallback(
    (_: unknown, params: { height: number; width: number }) => {
      updateLayout({ height: params.height, width: params.width });
    },
    [updateLayout],
  );

  return (
    <div
      className="group/node relative h-full w-full"
      onMouseEnter={(e) => {
        setNavigatingNode(id);
        useUiStore.getState().setLastMousePos({ x: e.clientX, y: e.clientY });
      }}
      onMouseLeave={() => {
        resetNavigatingNode(id);
      }}
    >
      <NodeJumpOverlay id={id} isEditing={isEditing} />

      <NodeShell
        autoHeight={!typedData.presentation?.height}
        className={cn("min-w-[250px]", isUser ? "bg-primary/10" : "bg-card")}
        nodeId={id}
        onDoubleClick={() => {
          setIsEditing(true);
        }}
        selected={selected}
      >
        <NodeResizer
          color="var(--primary-color)"
          handleStyle={{
            backgroundColor: "var(--primary-color)",
            border: "2px solid white",
            borderRadius: "50%",
            height: 10,
            width: 10,
          }}
          isVisible={selected}
          minHeight={100}
          minWidth={250}
          onResizeEnd={handleResizeEnd}
        />

        <Handle className="!w-2 !h-2 !bg-primary-color" position={Position.Top} type="target" />

        <div className="flex flex-col gap-1 flex-1 px-4 pt-3 pb-4 h-full">
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

          <div className="flex-1 overflow-auto">
            <MarkdownRenderer
              content={content || ""}
              isEditing={isEditing}
              onEdit={setContent}
              onToggleEditing={setIsEditing}
            />
          </div>
        </div>

        <Handle
          className="!w-2 !h-2 !bg-primary-color"
          position={Position.Bottom}
          style={{ bottom: -5 }}
          type="source"
        />
      </NodeShell>
    </div>
  );
});

ChatMessageNode.displayName = "ChatMessageNode";
