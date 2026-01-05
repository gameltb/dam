import React, { useState } from "react";
import { type Node as RFNode } from "@xyflow/react";
import { RenderMode } from "../../generated/flowcraft/v1/core/node_pb";
import type { DynamicNodeData } from "../../types";
import { NodeInfoPanel } from "./NodeInfoPanel";
import { cn } from "../../lib/utils";

export interface BaseNodeProps<T extends RFNode> {
  id: string;
  data: T["data"];
  type?: string;
  selected?: boolean;
  width?: number;
  height?: number;
  x?: number;
  y?: number;
  measured?: { width?: number; height?: number };
  style?: React.CSSProperties;
  initialMode?: RenderMode;
  renderMedia?: React.ComponentType<{
    id: string;
    data: T["data"];
    onOverflowChange: (overflow: "visible" | "hidden") => void;
  }>;
  renderWidgets?: React.ComponentType<{
    id: string;
    data: T["data"];
    onToggleMode: () => void;
  }>;
  renderChat?: React.ComponentType<{
    id: string;
    data: T["data"];
    updateNodeData: (nodeId: string, data: Partial<DynamicNodeData>) => void;
  }>;
  handles?: React.ReactNode;
  wrapperStyle?: React.CSSProperties;
  onOverflowChange?: (overflow: "visible" | "hidden") => void;
  updateNodeData?: (nodeId: string, data: Partial<DynamicNodeData>) => void;
}

export function BaseNode<T extends RFNode>({
  id,
  data,
  type,
  selected,
  measured,
  width,
  height,
  x,
  y,
  initialMode = RenderMode.MODE_WIDGETS,
  renderMedia: RenderMedia,
  renderWidgets: RenderWidgets,
  renderChat: RenderChat,
  handles,
  wrapperStyle,
  onOverflowChange,
  updateNodeData,
  ...rest
}: BaseNodeProps<T>) {
  const [internalMode, setInternalMode] = useState<RenderMode>(initialMode);
  // Default to visible so handles are never cut off
  const [overflow, setOverflow] = useState<"visible" | "hidden">("visible");

  const mode = (data as DynamicNodeData).activeMode ?? internalMode;
  const isMedia = mode === RenderMode.MODE_MEDIA;
  const isChat = mode === RenderMode.MODE_CHAT;
  const nodeType = type ?? "node";

  const toggleMode = () => {
    const nextMode =
      mode === RenderMode.MODE_WIDGETS
        ? RenderMode.MODE_MEDIA
        : RenderMode.MODE_WIDGETS;
    const dynamicData = data as unknown as DynamicNodeData;
    if (typeof dynamicData.onChange === "function") {
      (
        dynamicData.onChange as (
          id: string,
          data: { activeMode: RenderMode },
        ) => void
      )(id, {
        activeMode: nextMode,
      });
    } else {
      setInternalMode(nextMode);
    }
  };

  const handleOverflowChange = (newOverflow: "visible" | "hidden") => {
    setOverflow(newOverflow);
    onOverflowChange?.(newOverflow);
  };

  return (
    <>
      {selected && (
        <NodeInfoPanel
          nodeId={id}
          templateId={nodeType}
          width={measured?.width ?? width ?? 0}
          height={measured?.height ?? height ?? 0}
          x={x ?? 0}
          y={y ?? 0}
        />
      )}
      <div
        className={cn(
          "w-full h-full flex flex-col box-border rounded-[inherit]",
          isMedia ? "p-0" : "p-0"
        )}
        style={{
          overflow: overflow,
          ...wrapperStyle,
        }}
      >
        {isMedia && RenderMedia && (
          <RenderMedia
            id={id}
            data={data as DynamicNodeData}
            {...rest}
            onOverflowChange={handleOverflowChange}
          />
        )}
        {isChat && RenderChat && updateNodeData && (
          <RenderChat
            id={id}
            data={data as DynamicNodeData}
            updateNodeData={updateNodeData}
          />
        )}
        {!isMedia && !isChat && RenderWidgets && (
          <RenderWidgets
            id={id}
            data={data as DynamicNodeData}
            {...rest}
            onToggleMode={toggleMode}
          />
        )}
      </div>
      {handles}
    </>
  );
}
