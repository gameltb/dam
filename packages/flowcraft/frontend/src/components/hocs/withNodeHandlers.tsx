import React from "react";
import { type Node, type NodeProps, NodeResizer } from "@xyflow/react";
import { BaseNode } from "../base/BaseNode";
import type { DynamicNodeData } from "../../types";
import { MediaType } from "../../types";
import { useNodeLayout } from "../../hooks/useNodeLayout";

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

    const { minHeight, isMedia } = useNodeLayout(data);

    const shouldLockAspectRatio =
      isMedia &&
      (data.media?.type === MediaType.MEDIA_IMAGE ||
        data.media?.type === MediaType.MEDIA_VIDEO);

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
              TYPE: {(data.typeId ?? type).toUpperCase()}
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
          minHeight={minHeight}
          keepAspectRatio={shouldLockAspectRatio}
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
