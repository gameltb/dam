import { create } from "@bufbuild/protobuf";
import { Position, Handle as ReactFlowHandle, useStore } from "@xyflow/react";
import React, { useState } from "react";

import { PortMainType } from "@/generated/flowcraft/v1/core/base_pb";
import {
  PortSchema,
  PortStyle,
  PortTypeSchema,
} from "@/generated/flowcraft/v1/core/node_pb";
import { useUiStore } from "@/store/uiStore";
import { getValidator, validateConnection } from "@/utils/portValidators";
import { PORT_MAIN_TYPE_FROM_PROTO } from "@/utils/protoAdapter";

import { PortIcon } from "./PortIcon";

interface PortHandleProps {
  color?: string;
  description?: string;
  isGeneric?: boolean;
  isImplicit?: boolean;
  isPresentation?: boolean;
  itemType?: string;
  label?: string;
  mainType?: PortMainType;
  nodeId: string;
  portId: string;
  sideOffset?: number;
  style?: PortStyle;
  type: "source" | "target";
}

export const PortHandle: React.FC<PortHandleProps> = ({
  color = "var(--primary-color)",
  description,
  isGeneric = false,
  isImplicit = false,
  isPresentation = false,
  itemType = "",
  mainType = PortMainType.ANY,
  nodeId,
  portId,
  sideOffset = 0,
  style = PortStyle.CIRCLE,
  type,
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
    create(PortTypeSchema, { isGeneric, itemType, mainType }),
  );
  const inputCount = edges.filter(
    (e) => e.target === nodeId && e.targetHandle === portId,
  ).length;

  // --- Dynamic Guarding & Tooltip Logic ---
  let isInvalidTarget = false;
  let validationResult: null | { canConnect: boolean; reason?: string } = null;

  if (activeConnection) {
    if (activeConnection.type === type || activeConnection.nodeId === nodeId) {
      isInvalidTarget = true;
    } else {
      // Perform validation for tooltip feedback
      const sourcePort = create(PortSchema, {
        color: "",
        description: "",
        id: activeConnection.handleId,
        label: "",
        style: PortStyle.CIRCLE,
        type: {
          isGeneric: false,
          itemType: activeConnection.itemType,
          mainType: activeConnection.mainType,
        },
      });
      const targetPort = create(PortSchema, {
        color: "",
        description: "",
        id: portId,
        label: "",
        style: style,
        type: {
          isGeneric: isGeneric,
          itemType: itemType,
          mainType: mainType,
        },
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

  const tooltip = `Type: ${PORT_MAIN_TYPE_FROM_PROTO[mainType] ?? "any"}${itemType ? `<${itemType}>` : ""}\nLimit: ${validator.getMaxInputs() === 999 ? "Multiple" : "Single"}\n${description ?? ""}`;

  // --- Presentation Mode Specific Styles ---
  const trianglePath = isLeft
    ? "M 7 0 L 0 7 L 7 14 Z"
    : "M 7 0 L 14 7 L 7 14 Z";

  const handleStyle: React.CSSProperties = isPresentation
    ? {
        alignItems: "center",
        background: "transparent",
        border: "none",
        boxSizing: "border-box",
        display: "flex",
        height: "14px",
        [isLeft ? "left" : "right"]: 0,
        justifyContent: "center",
        minHeight: "14px",
        minWidth: "14px",
        opacity: activeConnection || isConnected || isHovered ? 1 : 0.6,
        padding: 0,
        pointerEvents: activeConnection && isInvalidTarget ? "none" : "auto",
        position: "absolute",
        top: "50%",
        transform: isLeft ? "translate(-50%, -50%)" : "translate(50%, -50%)",
        transition: "all 0.2s ease",
        width: "14px",
        zIndex: 10,
      }
    : {
        alignItems: "center",
        background: "transparent",
        border: "none",
        boxSizing: "border-box",
        display: "flex",
        filter: activeConnection && isInvalidTarget ? "grayscale(1)" : "none",
        height: "10px",
        [isLeft ? "left" : "right"]: -sideOffset,
        justifyContent: "center",
        minHeight: "10px",
        minWidth: "10px",
        opacity:
          isImplicit && !activeConnection && !isConnected
            ? 0.001
            : activeConnection && isInvalidTarget
              ? 0.15
              : 1,
        padding: 0,
        pointerEvents: activeConnection && isInvalidTarget ? "none" : "auto",
        position: "absolute",
        top: "50%",
        transform: isLeft ? "translate(-50%, -50%)" : "translate(50%, -50%)",
        transition: "all 0.2s ease",
        width: "10px",
        zIndex: 10,
      };

  return (
    <ReactFlowHandle
      id={portId}
      isConnectable={isConnectable}
      onMouseEnter={() => {
        setIsHovered(true);
      }}
      onMouseLeave={() => {
        setIsHovered(false);
      }}
      position={isLeft ? Position.Left : Position.Right}
      style={handleStyle}
      title={activeConnection ? undefined : tooltip}
      type={type}
    >
      {isPresentation ? (
        <svg
          height="14"
          style={{ overflow: "visible" }}
          viewBox="0 0 14 14"
          width="14"
        >
          <path
            d={trianglePath}
            fill={color}
            fillOpacity={isConnected ? 0.6 : 0.2}
            stroke={color}
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth="1.5"
            style={{ transition: "all 0.2s ease" }}
          />
        </svg>
      ) : (
        <div
          style={{
            alignItems: "center",
            display: "flex",
            height: "100%",
            justifyContent: "center",
            transform: `scale(${String(isImplicit && !activeConnection && !isConnected ? 0.5 : 1)})`,
            transition: "transform 0.1s ease",
            width: "100%",
          }}
        >
          <PortIcon
            color={color}
            isConnected={isConnected}
            mainType={PORT_MAIN_TYPE_FROM_PROTO[mainType]}
            style={style}
          />
        </div>
      )}

      {isHovered && activeConnection && (
        <div
          style={{
            backgroundColor: "var(--panel-bg)",
            border: `1px solid ${validationResult?.canConnect ? "#4ade80" : "var(--node-border)"}`,
            borderRadius: "4px",
            boxShadow: "0 4px 12px rgba(0,0,0,0.3)",
            display: "flex",
            flexDirection: "column",
            fontSize: "10px",
            gap: "2px",
            [isLeft ? "left" : "right"]: 20,
            padding: "4px 8px",
            pointerEvents: "none",
            position: "absolute",
            top: "50%",
            transform: "translateY(-50%)",
            whiteSpace: "nowrap",
            zIndex: 100,
          }}
        >
          <div style={{ color: "var(--sub-text)" }}>
            Connection: {PORT_MAIN_TYPE_FROM_PROTO[activeConnection.mainType]}
          </div>
          <div
            style={{
              color: validationResult?.canConnect ? "#4ade80" : "inherit",
              fontWeight: "bold",
            }}
          >
            This Port: {PORT_MAIN_TYPE_FROM_PROTO[mainType]}
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
