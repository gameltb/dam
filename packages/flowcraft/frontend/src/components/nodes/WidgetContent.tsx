import React, { memo } from "react";
import { flowcraft_proto } from "../../generated/flowcraft_proto";
import type { WidgetDef, DynamicNodeData } from "../../types";
import { WidgetWrapper } from "../widgets/WidgetWrapper";
import { TextField } from "../widgets/TextField";
import { SelectField } from "../widgets/SelectField";
import { CheckboxField } from "../widgets/CheckboxField";
import { SliderField } from "../widgets/SliderField";
import { PortHandle } from "../base/PortHandle";
import { useMockSocket } from "../../hooks/useMockSocket";
import { NodeLabel } from "./NodeLabel";
import { PortLabelRow } from "./PortLabelRow";

const _WidgetType = flowcraft_proto.v1.WidgetType;
const PortStyle = flowcraft_proto.v1.PortStyle;

interface WidgetRendererProps {
  nodeId: string;
  widget: WidgetDef;
  onValueChange: (val: unknown) => void;
  onClick: () => void;
}

const WidgetRenderer: React.FC<WidgetRendererProps> = memo(
  ({ nodeId, widget, onValueChange, onClick }) => {
    const { sendWidgetUpdate } = useMockSocket({ disablePolling: true });

    const handleValueChange = (val: unknown) => {
      onValueChange(val);
      sendWidgetUpdate(nodeId, widget.id, val);
    };

    let component;
    switch (widget.type) {
      case _WidgetType.WIDGET_TEXT:
        component = (
          <TextField
            value={widget.value as string}
            onChange={handleValueChange}
            label={widget.label}
          />
        );
        break;
      case _WidgetType.WIDGET_SELECT:
        component = (
          <SelectField
            value={widget.value}
            onChange={handleValueChange}
            label={widget.label}
            options={widget.options ?? []}
            onFetchOptions={() => Promise.resolve([])}
          />
        );
        break;
      case _WidgetType.WIDGET_CHECKBOX:
        component = (
          <CheckboxField
            value={!!widget.value}
            onChange={handleValueChange}
            label={widget.label}
          />
        );
        break;
      case _WidgetType.WIDGET_SLIDER:
        component = (
          <SliderField
            value={widget.value as number}
            onChange={handleValueChange}
            label={widget.label}
            min={(widget.config?.min as number | undefined) ?? 0}
            max={(widget.config?.max as number | undefined) ?? 100}
          />
        );
        break;
      case _WidgetType.WIDGET_BUTTON:
        component = (
          <button
            className="nodrag"
            onClick={onClick}
            style={{ width: "100%", padding: "4px" }}
          >
            {widget.label}
          </button>
        );
        break;
      default:
        return null;
    }
    return <div>{component}</div>;
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
            data.onChange(nodeId, { label });
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
          {data.widgets?.map((w) => (
            <div key={w.id} style={{ position: "relative", width: "100%" }}>
              {w.inputPortId && (
                <PortHandle
                  nodeId={id}
                  portId={w.inputPortId}
                  type="target"
                  sideOffset={12}
                  style={PortStyle.PORT_STYLE_CIRCLE}
                  color="var(--primary-color)"
                />
              )}

              <WidgetWrapper
                isSwitchable={isSwitchable}
                onToggleMode={onToggleMode}
                inputPortId={w.inputPortId}
                nodeId={id}
              >
                <WidgetRenderer
                  nodeId={id}
                  widget={w}
                  onValueChange={(val) => {
                    const updatedWidgets = (data.widgets ?? []).map((item) =>
                      item.id === w.id ? { ...item, value: val } : item,
                    );

                    data.onChange(id, { widgets: updatedWidgets });
                  }}
                  onClick={() => {
                    data.onWidgetClick?.(id, w.id);
                  }}
                />
              </WidgetWrapper>
            </div>
          ))}
        </div>
      </div>
    );
  },
);
