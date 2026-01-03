import React from "react";
import { WidgetType } from "../../generated/flowcraft/v1/core/node_pb";
import { TextField } from "./TextField";
import { SelectField } from "./SelectField";
import { CheckboxField } from "./CheckboxField";
import { SliderField } from "./SliderField";

export interface WidgetRendererProps {
  nodeId: string;
  value: unknown;
  label: string;
  options?: { label: string; value: unknown }[];
  config?: Record<string, unknown>;
  onChange: (val: unknown) => void;
  onClick: () => void;
}

export const WIDGET_COMPONENTS: Record<
  number,
  React.ComponentType<WidgetRendererProps>
> = {
  [WidgetType.WIDGET_TEXT]: ({ value, onChange, label }) => (
    <TextField value={value as string} onChange={onChange} label={label} />
  ),
  [WidgetType.WIDGET_SELECT]: ({ value, onChange, label, options }) => (
    <SelectField
      value={value}
      onChange={onChange}
      label={label}
      options={options ?? []}
      onFetchOptions={() => Promise.resolve([])}
    />
  ),
  [WidgetType.WIDGET_CHECKBOX]: ({ value, onChange, label }) => (
    <CheckboxField value={!!value} onChange={onChange} label={label} />
  ),
  [WidgetType.WIDGET_SLIDER]: ({ value, onChange, label, config }) => (
    <SliderField
      value={value as number}
      onChange={onChange}
      label={label}
      min={(config?.min as number | undefined) ?? 0}
      max={(config?.max as number | undefined) ?? 100}
    />
  ),
  [WidgetType.WIDGET_BUTTON]: ({ label, onClick }) => (
    <button
      className="nodrag"
      onClick={onClick}
      style={{ width: "100%", padding: "4px" }}
    >
      {label}
    </button>
  ),
};
