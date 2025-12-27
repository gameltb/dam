import React, { memo } from "react";
import { useNodeConnections } from "@xyflow/react";
import { PortHandle } from "../base/PortHandle";
import { flowcraft_proto } from "../../generated/flowcraft_proto";

interface PortLabelRowProps {
  nodeId: string;
  inputPort?: flowcraft_proto.v1.IPort;
  outputPort?: flowcraft_proto.v1.IPort;
}

export const PortLabelRow: React.FC<PortLabelRowProps> = memo(
  ({ nodeId, inputPort, outputPort }) => {
    const inputConnections = useNodeConnections({
      handleType: "target",
      handleId: inputPort?.id ?? undefined,
    });
    const outputConnections = useNodeConnections({
      handleType: "source",
      handleId: outputPort?.id ?? undefined,
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
                portId={inputPort.id ?? ""}
                type="target"
                style={inputPort.style ?? undefined}
                mainType={inputPort.type?.mainType ?? undefined}
                itemType={inputPort.type?.itemType ?? undefined}
                isGeneric={!!inputPort.type?.isGeneric}
                color={inputPort.color ?? undefined}
                description={inputPort.description ?? undefined}
                sideOffset={12}
              />
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
              <PortHandle
                nodeId={nodeId}
                portId={outputPort.id ?? ""}
                type="source"
                style={outputPort.style ?? undefined}
                mainType={outputPort.type?.mainType ?? undefined}
                itemType={outputPort.type?.itemType ?? undefined}
                isGeneric={!!outputPort.type?.isGeneric}
                color={outputPort.color ?? undefined}
                description={outputPort.description ?? undefined}
                sideOffset={12}
              />
            </>
          )}{" "}
        </div>
      </div>
    );
  },
);
