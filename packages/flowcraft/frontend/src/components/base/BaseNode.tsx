import React, { useState } from "react";
import { type Node as RFNode } from "@xyflow/react";
import { RenderMode } from "../../generated/flowcraft/v1/node_pb";
import type { DynamicNodeData } from "../../types";

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
  handles?: React.ReactNode;
  wrapperStyle?: React.CSSProperties;
  onOverflowChange?: (overflow: "visible" | "hidden") => void;
}

const NodeInfoPanel = ({
  nodeId,
  templateId,
  width,
  height,
  x,
  y,
}: {
  nodeId: string;
  templateId: string;
  width: number;
  height: number;
  x: number;
  y: number;
}) => (
  <div
    className="nodrag nopan"
    style={{
      position: "absolute",
      top: "-48px",
      left: "50%",
      transform: "translateX(-50%)",
      backgroundColor: "rgba(30, 30, 30, 0.95)",
      backdropFilter: "blur(12px)",
      border: "1px solid var(--primary-color)",
      borderRadius: "8px",
      padding: "6px 14px",
      display: "flex",
      gap: "14px",
      alignItems: "center",
      boxShadow:
        "0 8px 25px rgba(0,0,0,0.6), 0 0 0 1px rgba(100, 108, 255, 0.2)",
      zIndex: 9999,
      pointerEvents: "none",
      whiteSpace: "nowrap",
      fontSize: "11px",
      fontWeight: 600,
      color: "#fff",
      animation: "nodeInfoFadeIn 0.2s cubic-bezier(0.16, 1, 0.3, 1)",
    }}
  >
    <style>
      {`
          @keyframes nodeInfoFadeIn {
            from { opacity: 0; transform: translate(-50%, 5px); }
            to { opacity: 1; transform: translate(-50%, 0); }
          }
        `}
    </style>
    <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
      <span style={{ opacity: 0.5, fontSize: "9px" }}>NODE_ID</span>
      <span>{nodeId.slice(0, 8)}</span>
    </div>
    <div
      style={{
        width: "1px",
        height: "12px",
        backgroundColor: "rgba(255,255,255,0.15)",
      }}
    />
    <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
      <span style={{ opacity: 0.5, fontSize: "9px" }}>TEMPLATE_ID</span>
      <span style={{ color: "var(--primary-color)", letterSpacing: "0.5px" }}>
        {templateId.toUpperCase()}
      </span>
    </div>
    <div
      style={{
        width: "1px",
        height: "12px",
        backgroundColor: "rgba(255,255,255,0.15)",
      }}
    />
    <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
      <div style={{ display: "flex", alignItems: "center", gap: "4px" }}>
        <span style={{ opacity: 0.5, fontSize: "9px" }}>POS</span>
        <span style={{ color: "#4caf50" }}>X:</span>
        <span style={{ minWidth: "25px" }}>{Math.round(x)}</span>
        <span style={{ color: "#4caf50" }}>Y:</span>
        <span style={{ minWidth: "25px" }}>{Math.round(y)}</span>
      </div>
      <div
        style={{
          width: "1px",
          height: "10px",
          backgroundColor: "rgba(255,255,255,0.1)",
        }}
      />
      <div style={{ display: "flex", alignItems: "center", gap: "4px" }}>
        <span style={{ opacity: 0.5, fontSize: "9px" }}>SIZE</span>
        <span>
          {Math.round(width)} × {Math.round(height)}
        </span>
      </div>
    </div>
  </div>
);

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
  handles,
  wrapperStyle,
  onOverflowChange,
  ...rest
}: BaseNodeProps<T>) {
  const [internalMode, setInternalMode] = useState<RenderMode>(initialMode);
  // Default to visible so handles are never cut off
  const [overflow, setOverflow] = useState<"visible" | "hidden">("visible");

  const mode = (data as DynamicNodeData).activeMode ?? internalMode;
  const isMedia = mode === RenderMode.MODE_MEDIA;
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
        style={{
          width: "100%",
          height: "100%",
          // Remove horizontal padding from here, move it to sub-components
          paddingTop: isMedia ? 0 : 0,
          paddingBottom: isMedia ? 0 : 0,
          overflow: overflow,
          display: "flex",
          flexDirection: "column",
          boxSizing: "border-box",
          borderRadius: "inherit",
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
        {!isMedia && RenderWidgets && (
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
