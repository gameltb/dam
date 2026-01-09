import { useNodeConnections } from "@xyflow/react";
import React, { memo } from "react";

import { type ClientPort } from "@/types";
import { getPortColor, getPortShape } from "@/utils/themeUtils";

import { PortHandle } from "../base/PortHandle";

interface PortLabelRowProps {
  inputPort?: ClientPort;
  nodeId: string;
  outputPort?: ClientPort;
}

export const PortLabelRow: React.FC<PortLabelRowProps> = memo(
  ({ inputPort, nodeId, outputPort }) => {
    const inputConnections = useNodeConnections({
      handleId: inputPort?.id,
      handleType: "target",
    });
    const outputConnections = useNodeConnections({
      handleId: outputPort?.id,
      handleType: "source",
    });

    const isInputConnected = inputConnections.length > 0;
    const isOutputConnected = outputConnections.length > 0;

    return (
      <div
        style={{
          alignItems: "center",
          boxSizing: "border-box",
          display: "flex",
          height: "24px",
          justifyContent: "space-between",
          padding: "0 12px",
          position: "relative",
          width: "100%",
        }}
      >
        <div
          style={{
            alignItems: "center",
            display: "flex",
            flex: 1,
            gap: "8px",
            height: "100%",
            minWidth: 0,
            position: "relative",
          }}
        >
          {inputPort && (
            <>
              <PortHandle
                color={getPortColor(inputPort.type)}
                description={inputPort.description}
                nodeId={nodeId}
                portId={inputPort.id}
                sideOffset={12}
                style={getPortShape(inputPort.type)}
                type="target"
              />
              {inputPort.label && (
                <div
                  style={{
                    color: "var(--text-color)",
                    fontSize: "11px",
                    fontWeight: 600,
                    opacity: isInputConnected ? 1 : 0.6,
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                  }}
                >
                  {inputPort.label}
                </div>
              )}
            </>
          )}
        </div>

        <div
          style={{
            alignItems: "center",
            display: "flex",
            flex: 1,
            gap: "8px",
            height: "100%",
            justifyContent: "flex-end",
            minWidth: 0,
            position: "relative",
            textAlign: "right",
          }}
        >
          {outputPort && (
            <>
              {outputPort.label && (
                <div
                  style={{
                    color: "var(--text-color)",
                    fontSize: "11px",
                    fontWeight: 600,
                    opacity: isOutputConnected ? 1 : 0.6,
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                  }}
                >
                  {outputPort.label}
                </div>
              )}
              <PortHandle
                color={getPortColor(outputPort.type)}
                description={outputPort.description}
                nodeId={nodeId}
                portId={outputPort.id}
                sideOffset={12}
                style={getPortShape(outputPort.type)}
                type="source"
              />
            </>
          )}{" "}
        </div>
      </div>
    );
  },
);
