import { type Node as RFNode } from "@xyflow/react";
import React, { useState } from "react";

import { RenderMode } from "@/generated/flowcraft/v1/core/node_pb";
import { cn } from "@/lib/utils";
import { type DynamicNodeData, OverflowMode } from "@/types";

import { NodeInfoPanel } from "./NodeInfoPanel";

export interface BaseNodeProps<T extends RFNode> {
  data: T["data"];
  handles?: React.ReactNode;
  height?: number;
  id: string;
  initialMode?: RenderMode;
  measured?: { height?: number; width?: number };
  onOverflowChange?: (overflow: OverflowMode) => void;
  renderChat?: React.ComponentType<{
    data: T["data"];
    id: string;
    updateNodeData: (nodeId: string, data: Partial<DynamicNodeData>) => void;
  }>;
  renderMedia?: React.ComponentType<{
    data: T["data"];
    id: string;
    onOverflowChange: (overflow: OverflowMode) => void;
  }>;
  renderWidgets?: React.ComponentType<{
    data: T["data"];
    id: string;
    onToggleMode: () => void;
  }>;
  selected?: boolean;
  style?: React.CSSProperties;
  type?: string;
  updateNodeData?: (nodeId: string, data: Partial<DynamicNodeData>) => void;
  width?: number;
  wrapperStyle?: React.CSSProperties;
  x?: number;
  y?: number;
}

export function BaseNode<T extends RFNode>({
  data,
  handles,
  height,
  id,
  initialMode = RenderMode.MODE_WIDGETS,
  measured,
  onOverflowChange,
  renderChat: RenderChat,
  renderMedia: RenderMedia,
  renderWidgets: RenderWidgets,
  selected,
  type,
  updateNodeData,
  width,
  wrapperStyle,
  x,
  y,
  ...rest
}: BaseNodeProps<T>) {
  const [internalMode, setInternalMode] = useState<RenderMode>(initialMode);
  // Default to visible so handles are never cut off
  const [overflow, setOverflow] = useState<OverflowMode>(OverflowMode.VISIBLE);

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

  const handleOverflowChange = (newOverflow: OverflowMode) => {
    setOverflow(newOverflow);
    onOverflowChange?.(newOverflow);
  };

  return (
    <>
      {selected && (
        <NodeInfoPanel
          height={measured?.height ?? height ?? 0}
          nodeId={id}
          templateId={nodeType}
          width={measured?.width ?? width ?? 0}
          x={x ?? 0}
          y={y ?? 0}
        />
      )}
      <div
        className={cn(
          "w-full h-full flex flex-col box-border rounded-[inherit]",
          isMedia ? "p-0" : "p-0",
        )}
        style={{
          overflow: overflow,
          ...wrapperStyle,
        }}
      >
        {isMedia && RenderMedia && (
          <RenderMedia
            data={data as DynamicNodeData}
            id={id}
            {...rest}
            onOverflowChange={handleOverflowChange}
          />
        )}
        {isChat && RenderChat && updateNodeData && (
          <RenderChat
            data={data as DynamicNodeData}
            id={id}
            updateNodeData={updateNodeData}
          />
        )}
        {!isMedia && !isChat && RenderWidgets && (
          <RenderWidgets
            data={data as DynamicNodeData}
            id={id}
            {...rest}
            onToggleMode={toggleMode}
          />
        )}
      </div>
      {handles}
    </>
  );
}
