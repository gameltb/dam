import React, { memo } from "react";
import { useNodeConnections } from "@xyflow/react";
import { PortHandle } from "../base/PortHandle";
import { type ClientPort } from "../../types";
import { getPortColor, getPortShape } from "../../utils/themeUtils";

interface PortLabelRowProps {
  nodeId: string;
  inputPort?: ClientPort;
  outputPort?: ClientPort;
}

export const PortLabelRow: React.FC<PortLabelRowProps> = memo(
  ({ nodeId, inputPort, outputPort }) => {
    const inputConnections = useNodeConnections({
      handleType: "target",
      handleId: inputPort?.id,
    });
    const outputConnections = useNodeConnections({
      handleType: "source",
      handleId: outputPort?.id,
    });

    const isInputConnected = inputConnections.length > 0;
    const isOutputConnected = outputConnections.length > 0;

    return (
      <div
        style={{
          display: "flex",
          alignItems: "center",
          height: "24px",
          position: "relative",
          width: "100%",
          boxSizing: "border-box",
          justifyContent: "space-between",
          padding: "0 12px",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "8px",
            flex: 1,
            minWidth: 0,
            position: "relative",
            height: "100%",
          }}
        >
          {inputPort && (
            <>
              <PortHandle
                nodeId={nodeId}
                portId={inputPort.id}
                type="target"
                style={getPortShape(inputPort.type)}
                color={getPortColor(inputPort.type)}
                description={inputPort.description}
                sideOffset={12}
              />
              {inputPort.label && (
                <div
                  style={{
                    fontSize: "11px",
                    fontWeight: 600,
                    color: "var(--text-color)",
                    whiteSpace: "nowrap",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    opacity: isInputConnected ? 1 : 0.6,
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
            display: "flex",
            alignItems: "center",
            gap: "8px",
            flex: 1,
            minWidth: 0,
            justifyContent: "flex-end",
            textAlign: "right",
            position: "relative",
            height: "100%",
          }}
        >
          {outputPort && (
            <>
              {outputPort.label && (
                <div
                  style={{
                    fontSize: "11px",
                    fontWeight: 600,
                    color: "var(--text-color)",
                    whiteSpace: "nowrap",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    opacity: isOutputConnected ? 1 : 0.6,
                  }}
                >
                  {outputPort.label}
                </div>
              )}
              <PortHandle
                nodeId={nodeId}
                portId={outputPort.id}
                type="source"
                style={getPortShape(outputPort.type)}
                color={getPortColor(outputPort.type)}
                description={outputPort.description}
                sideOffset={12}
              />
            </>
          )}{" "}
        </div>
      </div>
    );
  },
);
