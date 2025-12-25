import React from "react";
import { type Node, type NodeProps, NodeResizer } from "@xyflow/react";
import { BaseNode } from "../base/BaseNode";
import { type DynamicNodeData, RenderMode, MediaType } from "../../types";

export type NodeRendererProps<T extends Node> = NodeProps<T>;

export function withNodeHandlers<
  D extends DynamicNodeData = DynamicNodeData,
  T extends Node<D, string> = Node<D, string>,
>(
  RenderMedia: React.ComponentType<NodeRendererProps<T>>,
  RenderWidgets: React.ComponentType<
    NodeRendererProps<T> & { onToggleMode: () => void }
  >,
) {
  return function NodeWithHandlers(props: NodeProps<T>) {
    const { data, selected, type, positionAbsoluteX, positionAbsoluteY } =
      props;

    const isMedia = data.activeMode === RenderMode.MODE_MEDIA;
    const isAudio = isMedia && data.media?.type === MediaType.MEDIA_AUDIO;

    // --- Dynamic Min Height Calculation ---
    const HEADER_HEIGHT = 46;
    const PORT_HEIGHT =
      Math.max(data.inputPorts?.length || 0, data.outputPorts?.length || 0) *
      24;
    const WIDGETS_HEIGHT = (data.widgets?.length || 0) * 55;

    const calculatedMinHeight = isMedia
      ? isAudio
        ? 110
        : 50
      : HEADER_HEIGHT + PORT_HEIGHT + WIDGETS_HEIGHT + 20;

    return (
      <div
        className="custom-node"
        style={{
          position: "relative",
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          borderRadius: "8px",
          backgroundColor: "var(--node-bg)",
          color: "var(--text-color)",
          // Permanent thin border to maintain static layout
          border: "1px solid",
          borderColor: selected ? "var(--primary-color)" : "var(--node-border)",
          // Subtle drop shadow + Intense glow when selected
          boxShadow: selected
            ? "0 0 0 1px var(--primary-color), 0 0 15px rgba(100, 108, 255, 0.4), 0 10px 25px rgba(0,0,0,0.4)"
            : "0 4px 12px rgba(0,0,0,0.2)",
          boxSizing: "border-box",
          transition: "border-color 0.2s ease, box-shadow 0.2s ease",
        }}
      >
        <style>{`
            .custom-node:hover .react-flow__handle { opacity: 1 !important; }
        `}</style>

        {selected && (
          <div
            style={{
              position: "absolute",
              top: -22,
              left: 0,
              right: 0,
              display: "flex",
              justifyContent: "space-between",
              fontSize: "10px",
              color: "var(--sub-text)",
              pointerEvents: "none",
            }}
          >
            <div
              style={{
                background: "var(--panel-bg)",
                padding: "2px 6px",
                borderRadius: "4px 4px 0 0",
                border: "1px solid var(--primary-color)",
                borderBottom: "none",
              }}
            >
              TYPE: {String(type).toUpperCase()}
            </div>
            <div
              style={{
                background: "var(--panel-bg)",
                padding: "2px 6px",
                borderRadius: "4px 4px 0 0",
                border: "1px solid var(--primary-color)",
                borderBottom: "none",
              }}
            >
              {Math.round(positionAbsoluteX)},{Math.round(positionAbsoluteY)}
            </div>
          </div>
        )}

        <NodeResizer
          isVisible={selected}
          minWidth={180}
          minHeight={calculatedMinHeight}
          keepAspectRatio={isMedia}
          handleStyle={{
            width: 8,
            height: 8,
            borderRadius: "50%",
            background: "var(--primary-color)",
            border: "2px solid white",
          }}
        />

        <BaseNode<T>
          {...props}
          renderMedia={RenderMedia}
          renderWidgets={RenderWidgets}
          handles={null}
        />
      </div>
    );
  };
}
