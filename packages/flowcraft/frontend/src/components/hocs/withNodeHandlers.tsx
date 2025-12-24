import React, { useState } from "react";
import { type Node, type NodeProps, NodeResizer } from "@xyflow/react";
import { BaseNode } from "../base/BaseNode";
import { Handle } from "../base/Handle";
import { Position } from "@xyflow/react";
import { type DynamicNodeData } from "../../types";

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
    const { id, data, selected } = props;
    const [isResizing, setIsResizing] = useState(false);
    const [isHovered, setIsHovered] = useState(false);

    const isMedia = data.activeMode === "media";

    return (
      <div
        className="custom-node"
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        style={{
          position: "relative",
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          borderRadius: "8px",
          backgroundColor: "var(--node-bg, #2a2a2a)",
          border: selected
            ? "2px solid #646cff"
            : "1px solid rgba(255,255,255,0.1)",
          boxShadow: "0 4px 12px rgba(0,0,0,0.2)",
          boxSizing: "border-box",
          transition: "border-color 0.2s, box-shadow 0.2s",
        }}
        data-testid={`${props.type}-node-${id}`}
      >
        <NodeResizer
          isVisible={selected}
          minWidth={isMedia ? 50 : 150}
          minHeight={isMedia ? 50 : 80}
          keepAspectRatio={isMedia}
          lineStyle={{
            border: isResizing ? "1px solid #646cff" : "none",
          }}
          handleStyle={{
            opacity: isResizing || isHovered ? 1 : 0,
            width: 8,
            height: 8,
            backgroundColor: "#646cff",
            border: "1px solid white",
          }}
          onResizeStart={() => setIsResizing(true)}
          onResizeEnd={() => setIsResizing(false)}
        />

        {/* User Handles (Visible, Centered) */}
        <Handle type="target" position={Position.Left} id="default-target" />

        <BaseNode<T>
          {...props}
          renderMedia={(baseProps) => <RenderMedia {...props} {...baseProps} />}
          renderWidgets={(baseProps) => (
            <RenderWidgets {...props} {...baseProps} />
          )}
        />

        <Handle type="source" position={Position.Right} id="default-source" />

        {/* System Handles (Hidden, Top-Left/Top-Right, Code-only) */}
        <Handle
          type="target"
          position={Position.Left}
          id="system-target"
          isConnectable={false}
          style={{ top: 15, opacity: 0, pointerEvents: "none" }}
        />
        <Handle
          type="source"
          position={Position.Right}
          id="system-source"
          isConnectable={false}
          style={{ top: 15, opacity: 0, pointerEvents: "none" }}
        />
      </div>
    );
  };
}
