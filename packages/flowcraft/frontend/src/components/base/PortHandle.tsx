import React, { useState } from "react";
import { Handle as ReactFlowHandle, Position, useStore } from "@xyflow/react";
import {
  PortStyle,
  PortSchema,
  PortTypeSchema,
} from "../../generated/core/node_pb";
import { getValidator, validateConnection } from "../../utils/portValidators";
import { useUiStore } from "../../store/uiStore";
import { create } from "@bufbuild/protobuf";

interface PortHandleProps {
  nodeId: string;
  portId: string;
  type: "source" | "target";
  style?: PortStyle;
  mainType?: string;
  itemType?: string;
  isGeneric?: boolean;
  color?: string;
  label?: string;
  description?: string;
  sideOffset?: number;
  isImplicit?: boolean;
  isPresentation?: boolean;
}

const PortIcon: React.FC<{
  style: PortStyle;
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
    case PortStyle.SQUARE:
      return <div style={{ ...baseStyle, borderRadius: "2px" }} />;
    case PortStyle.DIAMOND:
      return (
        <div
          style={{
            ...baseStyle,
            transform: "rotate(45deg) scale(0.8)",
            borderRadius: "1px",
          }}
        />
      );
    case PortStyle.DASH:
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
  style = PortStyle.CIRCLE,
  mainType = "any",
  itemType = "",
  isGeneric = false,
  color = "var(--primary-color)",
  description,
  sideOffset = 0,
  isImplicit = false,
  isPresentation = false,
}) => {
  const isLeft = type === "target";
  const edges = useStore((s) => s.edges);
  const activeConnection = useUiStore((s) => s.connectionStartHandle);
  const [isHovered, setIsHovered] = useState(false);

  // Connection state
  const isConnected = edges.some((e) =>
    isLeft
      ? e.target === nodeId && e.targetHandle === portId
      : e.source === nodeId && e.sourceHandle === portId,
  );

  const validator = getValidator(
    create(PortTypeSchema, { mainType, itemType, isGeneric }),
  );
  const inputCount = edges.filter(
    (e) => e.target === nodeId && e.targetHandle === portId,
  ).length;

  // --- Dynamic Guarding & Tooltip Logic ---
  let isInvalidTarget = false;
  let validationResult: { canConnect: boolean; reason?: string } | null = null;

  if (activeConnection) {
    if (activeConnection.type === type || activeConnection.nodeId === nodeId) {
      isInvalidTarget = true;
    } else {
      // Perform validation for tooltip feedback
      const sourcePort = create(PortSchema, {
        id: activeConnection.handleId,
        type: {
          mainType: activeConnection.mainType,
          itemType: activeConnection.itemType,
          isGeneric: false,
        },
        label: "",
        color: "",
        style: PortStyle.CIRCLE,
        description: "",
      });
      const targetPort = create(PortSchema, {
        id: portId,
        type: {
          mainType: mainType,
          itemType: itemType,
          isGeneric: isGeneric,
        },
        label: "",
        color: "",
        style: style,
        description: "",
      });

      const res = validateConnection(
        { ...sourcePort, nodeId: activeConnection.nodeId },
        { ...targetPort, nodeId },
        edges,
      );
      validationResult = res;
      if (!res.canConnect) isInvalidTarget = true;
    }
  }

  const maxInputs = validator.getMaxInputs();
  const isConnectable =
    !isInvalidTarget && (!isLeft || inputCount < maxInputs || maxInputs === 1);

  const tooltip = `Type: ${mainType}${itemType ? `<${itemType}>` : ""}\nLimit: ${validator.getMaxInputs() === 999 ? "Multiple" : "Single"}\n${description ?? ""}`;

  // --- Presentation Mode Specific Styles ---

  // Left side (Target): Flat edge at center (7), Tip at x=0 (outside)

  // Right side (Source): Flat edge at center (7), Tip at x=14 (outside)

  // We use a 7px wide triangle so the base sits exactly on the handle's center line.

  const trianglePath = isLeft
    ? "M 7 0 L 0 7 L 7 14 Z"
    : "M 7 0 L 14 7 L 7 14 Z";

  const handleStyle: React.CSSProperties = isPresentation
    ? {
        position: "absolute",

        [isLeft ? "left" : "right"]: 0,

        top: "50%",

        width: "14px",

        height: "14px",

        background: "transparent",

        border: "none",

        minWidth: "14px",

        minHeight: "14px",

        padding: 0,

        boxSizing: "border-box",

        display: "flex",

        alignItems: "center",

        justifyContent: "center",

        zIndex: 10,

        pointerEvents: activeConnection && isInvalidTarget ? "none" : "auto",

        // Semi-transparent when idle, opaque when active/connected

        opacity: activeConnection || isConnected || isHovered ? 1 : 0.6,

        // Align handle center exactly to the node boundary

        transform: isLeft ? "translate(-50%, -50%)" : "translate(50%, -50%)",

        transition: "all 0.2s ease",
      }
    : {
        position: "absolute",
        [isLeft ? "left" : "right"]: -sideOffset,
        top: "50%",
        width: "10px",
        height: "10px",
        background: "transparent",
        border: "none",
        minWidth: "10px",
        minHeight: "10px",
        padding: 0,
        boxSizing: "border-box",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 10,
        pointerEvents: activeConnection && isInvalidTarget ? "none" : "auto",
        opacity:
          isImplicit && !activeConnection && !isConnected
            ? 0.001
            : activeConnection && isInvalidTarget
              ? 0.15
              : 1,
        transform: isLeft ? "translate(-50%, -50%)" : "translate(50%, -50%)",
        filter: activeConnection && isInvalidTarget ? "grayscale(1)" : "none",
        transition: "all 0.2s ease",
      };

  return (
    <ReactFlowHandle
      type={type}
      position={isLeft ? Position.Left : Position.Right}
      id={portId}
      isConnectable={isConnectable}
      title={activeConnection ? undefined : tooltip}
      onMouseEnter={() => {
        setIsHovered(true);
      }}
      onMouseLeave={() => {
        setIsHovered(false);
      }}
      style={handleStyle}
    >
      {isPresentation ? (
        <svg
          width="14"
          height="14"
          viewBox="0 0 14 14"
          style={{ overflow: "visible" }}
        >
          <path
            d={trianglePath}
            fill={color}
            fillOpacity={isConnected ? 0.6 : 0.2}
            stroke={color}
            strokeWidth="1.5"
            strokeLinejoin="round"
            strokeLinecap="round"
            style={{ transition: "all 0.2s ease" }}
          />
        </svg>
      ) : (
        <div
          style={{
            width: "100%",
            height: "100%",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            transform: `scale(${String(isImplicit && !activeConnection && !isConnected ? 0.5 : 1)})`,
            transition: "transform 0.1s ease",
          }}
        >
          <PortIcon
            style={style}
            mainType={mainType}
            color={color}
            isConnected={isConnected}
          />
        </div>
      )}

      {isHovered && activeConnection && (
        <div
          style={{
            position: "absolute",
            [isLeft ? "left" : "right"]: 20,
            top: "50%",
            transform: "translateY(-50%)",
            backgroundColor: "var(--panel-bg)",
            border: `1px solid ${validationResult?.canConnect ? "#4ade80" : "var(--node-border)"}`,
            borderRadius: "4px",
            padding: "4px 8px",
            whiteSpace: "nowrap",
            fontSize: "10px",
            boxShadow: "0 4px 12px rgba(0,0,0,0.3)",
            zIndex: 100,
            pointerEvents: "none",
            display: "flex",
            flexDirection: "column",
            gap: "2px",
          }}
        >
          <div style={{ color: "var(--sub-text)" }}>
            Connection: {activeConnection.mainType}
          </div>
          <div
            style={{
              color: validationResult?.canConnect ? "#4ade80" : "inherit",
              fontWeight: "bold",
            }}
          >
            This Port: {mainType}
          </div>
          {!validationResult?.canConnect && validationResult?.reason && (
            <div style={{ color: "#f87171", fontSize: "9px" }}>
              {validationResult.reason}
            </div>
          )}
        </div>
      )}
    </ReactFlowHandle>
  );
};
