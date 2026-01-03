import React, { memo } from "react";
import { PortStyle, type PortType } from "../../generated/flowcraft/v1/node_pb";
import { PortMainType } from "../../generated/flowcraft/v1/base_pb";
import type { WidgetDef, DynamicNodeData } from "../../types";
import { WidgetWrapper } from "../widgets/WidgetWrapper";
import { PortHandle } from "../base/PortHandle";
import { useFlowSocket } from "../../hooks/useFlowSocket";
import { NodeLabel } from "./NodeLabel";
import { PortLabelRow } from "./PortLabelRow";
import { useNodeHandlers } from "../../hooks/useNodeHandlers";
import { FlowcraftRJSF } from "../widgets/FlowcraftRJSF";

import { getPortColor } from "../../utils/themeUtils";

import { WIDGET_COMPONENTS } from "../widgets/widgetConfigs";

interface WidgetRendererProps {
  nodeId: string;
  widget: WidgetDef;
  onValueChange: (val: unknown) => void;
  onClick: () => void;
}

const WidgetRenderer: React.FC<WidgetRendererProps> = memo(
  ({ nodeId, widget, onValueChange, onClick }) => {
    const { updateWidget } = useFlowSocket({ disablePolling: true });

    const handleValueChange = (val: unknown) => {
      onValueChange(val);
      updateWidget(nodeId, widget.id, val);
    };

    const Component = WIDGET_COMPONENTS[widget.type];
    if (!Component) return null;

    return (
      <Component
        nodeId={nodeId}
        value={widget.value}
        label={widget.label}
        options={widget.options}
        config={widget.config}
        onChange={handleValueChange}
        onClick={onClick}
      />
    );
  },
);

interface WidgetContentProps {
  id: string;

  data: DynamicNodeData;

  selected?: boolean;

  onToggleMode: () => void;
}

export const WidgetContent: React.FC<WidgetContentProps> = memo(
  ({ id, data, selected, onToggleMode }) => {
    const { onChange, onWidgetClick } = useNodeHandlers(data, selected);
    const isSwitchable = data.modes.length > 1;

    const inputs = data.inputPorts ?? [];

    const outputs = data.outputPorts ?? [];

    const rowCount = Math.max(inputs.length, outputs.length);

    const portRows = [];

    for (let i = 0; i < rowCount; i++) {
      portRows.push({ input: inputs[i], output: outputs[i] });
    }

    return (
      <div
        style={{
          display: "flex",

          flexDirection: "column",

          minWidth: "150px",

          flex: 1,
        }}
      >
        <NodeLabel
          id={id}
          label={data.label}
          selected={selected}
          onChange={(nodeId, label) => {
            onChange(nodeId, { label });
          }}
        />

        <div
          style={{
            display: "flex",

            flexDirection: "column",
          }}
        >
          {portRows.map((row, idx) => (
            <PortLabelRow
              key={idx}
              nodeId={id}
              inputPort={row.input}
              outputPort={row.output}
            />
          ))}
        </div>

        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: "8px",
            padding: "8px 12px",
            position: "relative",
          }}
        >
          {/* Schema-Driven Widgets (RJSF) */}
          {data.widgetsSchemaJson && (
            <div className="nodrag nopan">
              <FlowcraftRJSF
                nodeId={id}
                schema={JSON.parse(data.widgetsSchemaJson)}
                formData={data.widgetsValues || {}}
                onChange={(newValues) => {
                  onChange(id, { widgetsValues: newValues });
                }}
              />
            </div>
          )}

          {/* Traditional Hardcoded Widgets */}
          {data.widgets?.map((w) => (
            <div key={w.id} style={{ position: "relative", width: "100%" }}>
              <WidgetWrapper
                isSwitchable={isSwitchable}
                onToggleMode={onToggleMode}
                inputPortId={w.inputPortId}
                nodeId={id}
              >
                <div style={{ position: "relative", width: "100%" }}>
                  {w.inputPortId && (
                    <PortHandle
                      nodeId={id}
                      portId={w.inputPortId}
                      type="target"
                      sideOffset={17}
                      style={PortStyle.CIRCLE}
                      color={getPortColor({
                        mainType: PortMainType.STRING,
                        itemType: "",
                        isGeneric: false,
                      } as unknown as PortType)}
                      isImplicit={true}
                    />
                  )}
                  <WidgetRenderer
                    nodeId={id}
                    widget={w}
                    onValueChange={(val) => {
                      const updatedWidgets = (data.widgets ?? []).map((item) =>
                        item.id === w.id ? { ...item, value: val } : item,
                      );

                      onChange(id, { widgets: updatedWidgets });
                    }}
                    onClick={() => {
                      onWidgetClick(id, w.id);
                    }}
                  />
                </div>
              </WidgetWrapper>
            </div>
          ))}
        </div>
      </div>
    );
  },
);
