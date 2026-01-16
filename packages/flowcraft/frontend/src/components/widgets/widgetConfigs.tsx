import React from "react";

import { WidgetType } from "@/generated/flowcraft/v1/core/node_pb";

import { CheckboxField } from "./CheckboxField";
import { SelectField } from "./SelectField";
import { SliderField } from "./SliderField";
import { TextField } from "./TextField";

export interface WidgetRendererProps {
  config?: Record<string, unknown>;
  label: string;
  nodeId: string;
  onChange: (val: unknown) => void;
  onClick: () => void;
  options?: { label: string; value: unknown }[];
  value: unknown;
}

export const WIDGET_COMPONENTS: Record<number, React.ComponentType<WidgetRendererProps>> = {
  [WidgetType.WIDGET_BUTTON]: ({ label, onClick }) => (
    <button className="nodrag" onClick={onClick} style={{ padding: "4px", width: "100%" }}>
      {label}
    </button>
  ),
  [WidgetType.WIDGET_CHECKBOX]: ({ label, onChange, value }) => (
    <CheckboxField label={label} onChange={onChange} value={!!value} />
  ),
  [WidgetType.WIDGET_SELECT]: ({ label, onChange, options, value }) => (
    <SelectField
      label={label}
      onChange={onChange}
      onFetchOptions={() => Promise.resolve([])}
      options={options ?? []}
      value={value}
    />
  ),
  [WidgetType.WIDGET_SLIDER]: ({ config, label, onChange, value }) => (
    <SliderField
      label={label}
      max={(config?.max as number | undefined) ?? 100}
      min={(config?.min as number | undefined) ?? 0}
      onChange={onChange}
      value={value as number}
    />
  ),
  [WidgetType.WIDGET_TEXT]: ({ label, onChange, value }) => (
    <TextField label={label} onChange={onChange} value={value as string} />
  ),
};
