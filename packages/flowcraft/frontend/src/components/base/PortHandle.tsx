import React from "react";
import { Handle as ReactFlowHandle, Position, useStore } from "@xyflow/react";
import {
  isDynamicNode,
  type AppNode,
  type PortStyle as PortStyleType,
} from "../../types";
import { flowcraft_proto } from "../../generated/flowcraft_proto";
import { getValidator } from "../../utils/portValidators";
import { useFlowStore } from "../../store/flowStore";

const PortStyle = flowcraft_proto.v1.PortStyle;

interface PortHandleProps {
  nodeId: string;
  portId: string;
  type: "source" | "target";
  style?: PortStyleType;
  mainType?: string;
  itemType?: string;
  isGeneric?: boolean;
  color?: string;
  label?: string;
  description?: string;
  sideOffset?: number;
}

const PortIcon: React.FC<{
  style: PortStyleType;
  mainType?: string;
  color: string;
  isConnected: boolean;
}> = ({ style, mainType, color, isConnected }) => {
  const baseStyle: React.CSSProperties = {
    width: "10px",
    height: "10px",
    background: isConnected ? color : "var(--node-bg)",
    border: `2px solid ${color}`,
    boxSizing: "border-box",
    transition: "all 0.2s ease",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    overflow: "hidden",
  };

  if (mainType === "list") {
    return (
      <div
        style={{
          ...baseStyle,
          borderRadius: "2px",
          flexDirection: "column",
          gap: "1px",
          padding: "1px",
        }}
      >
        <div
          style={{
            width: "100%",
            height: "2px",
            background: isConnected ? "white" : color,
            opacity: 0.8,
          }}
        />
        <div
          style={{
            width: "100%",
            height: "2px",
            background: isConnected ? "white" : color,
            opacity: 0.8,
          }}
        />
      </div>
    );
  }
  if (mainType === "set") {
    return (
      <div style={{ ...baseStyle, borderRadius: "50%" }}>
        <div
          style={{
            width: "4px",
            height: "4px",
            borderRadius: "50%",
            background: isConnected ? "white" : color,
          }}
        />
      </div>
    );
  }
  switch (style) {
    case PortStyle.PORT_STYLE_SQUARE:
      return <div style={{ ...baseStyle, borderRadius: "2px" }} />;
    case PortStyle.PORT_STYLE_DIAMOND:
      return (
        <div
          style={{
            ...baseStyle,
            transform: "rotate(45deg) scale(0.8)",
            borderRadius: "1px",
          }}
        />
      );
    case PortStyle.PORT_STYLE_DASH:
      return (
        <div
          style={{
            ...baseStyle,
            borderStyle: "dashed",
            borderRadius: "50%",
            background: "transparent",
          }}
        />
      );
    default:
      return <div style={{ ...baseStyle, borderRadius: "50%" }} />;
  }
};

export const PortHandle: React.FC<PortHandleProps> = ({
  nodeId,
  portId,
  type,
  style = PortStyle.PORT_STYLE_CIRCLE,
  mainType,
  itemType,
  isGeneric,
  color = "var(--primary-color)",
  description,
  sideOffset = 0,
}) => {
  const isLeft = type === "target";
  const edges = useStore((s) => s.edges);
  const nodes = useStore((s) => s.nodes);
  const activeConnection = useFlowStore((s) => s.connectionStartHandle);

  // Connection state
  const isConnected = edges.some((e) =>
    isLeft
      ? e.target === nodeId && e.targetHandle === portId
      : e.source === nodeId && e.sourceHandle === portId,
  );

  const validator = getValidator({ mainType, itemType, isGeneric });
  const inputCount = edges.filter(
    (e) => e.target === nodeId && e.targetHandle === portId,
  ).length;

  // --- Dynamic Guarding Logic ---
  let isInvalidTarget = false;

  if (activeConnection) {
    // 1. Directional Guard (Source must connect to Target)
    if (activeConnection.type === type) {
      isInvalidTarget = true;
    }
    // 2. Self-connection Guard
    else if (activeConnection.nodeId === nodeId) {
      isInvalidTarget = true;
    }
    // 3. Semantic & Capacity Guard
    else {
      const sourceNode = nodes.find((n) => n.id === activeConnection.nodeId);
      if (sourceNode && isDynamicNode(sourceNode as AppNode)) {
        const dynamicSourceNode = sourceNode as AppNode;
        const data = dynamicSourceNode.data as
          | flowcraft_proto.v1.INodeData
          | undefined;
        const sourcePort =
          data?.outputPorts?.find((p) => p.id === activeConnection.handleId) ??
          data?.inputPorts?.find((p) => p.id === activeConnection.handleId);

        if (sourcePort?.type) {
          const typeCompatible = validator.canAccept(sourcePort.type, {
            mainType,
            itemType,
            isGeneric,
          });
          const hasCapacity = inputCount < validator.getMaxInputs();

          if (!typeCompatible || (isLeft && !hasCapacity)) {
            isInvalidTarget = true;
          }
        }
      }
    }
  }

  // A port is connectable ONLY if it's a valid target for the current drag
  const isConnectable =
    !isInvalidTarget && (!isLeft || inputCount < validator.getMaxInputs());

  const tooltip = `Type: ${mainType ?? "any"}${itemType ? `<${itemType}>` : ""}\nLimit: ${validator.getMaxInputs() === 999 ? "Multiple" : "Single"}\n${description ?? ""}`;

  return (
    <ReactFlowHandle
      type={type}
      position={isLeft ? Position.Left : Position.Right}
      id={portId}
      isConnectable={isConnectable}
      title={tooltip}
      style={{
        position: "absolute",
        [isLeft ? "left" : "right"]: -sideOffset,
        top: "50%",
        transform: isLeft ? "translate(-50%, -50%)" : "translate(50%, -50%)",
        width: "14px",
        height: "14px",
        background: "transparent",
        border: "none",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 10,
        // Disable pointer events entirely for invalid targets to stop snapping
        pointerEvents: activeConnection && isInvalidTarget ? "none" : "auto",
        opacity: activeConnection && isInvalidTarget ? 0.15 : 1,
        filter: activeConnection && isInvalidTarget ? "grayscale(1)" : "none",
        transition: "all 0.2s ease",
      }}
    >
      <div style={{ width: "10px", height: "10px" }}>
        <PortIcon
          style={style}
          mainType={mainType}
          color={color}
          isConnected={isConnected}
        />
      </div>
    </ReactFlowHandle>
  );
};
