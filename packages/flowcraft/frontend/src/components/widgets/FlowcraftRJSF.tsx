import type { ObjectFieldTemplateProps, RegistryWidgetsType, RJSFSchema, UiSchema, WidgetProps } from "@rjsf/utils";

import { create } from "@bufbuild/protobuf";
import Form from "@rjsf/core";
import validator from "@rjsf/validator-ajv8";
import React, { useEffect, useState } from "react";

import { WidgetSignalSchema } from "@/generated/flowcraft/v1/core/signals_pb";
import { useFlowStore } from "@/store/flowStore";

/**
 * 自定义对象模板：移除 fieldset 和 legend
 */
const PlainObjectFieldTemplate = (props: ObjectFieldTemplateProps) => {
  return (
    <div style={{ width: "100%" }}>
      {props.properties.map((element) => (
        <div
          key={element.name}
          style={{
            marginBottom: element.name === "conversation" ? 0 : "10px",
            width: "100%",
          }}
        >
          {element.content}
        </div>
      ))}
    </div>
  );
};

import { useTable } from "spacetimedb/react";

import { tables } from "@/generated/spacetime";

const StreamingTextWidget = (props: WidgetProps) => {
  const { id, label, onChange } = props;
  const registry = props.registry;
  const value = props.value as string | undefined;
  const formContext = registry.formContext as Record<string, unknown> | undefined;
  const nodeId = formContext?.nodeId as string | undefined;
  const widgetId = id.split("_").pop();

  const [stWidgetValues] = useTable(tables.widgetValues);

  const streamedValue = React.useMemo(() => {
    if (!nodeId || !widgetId) return value ?? "";
    const entry = stWidgetValues.find((wv) => wv.nodeId === nodeId && wv.widgetId === widgetId);
    if (entry) {
      try {
        return JSON.parse(entry.value) as string;
      } catch {
        return entry.value;
      }
    }
    return value ?? "";
  }, [stWidgetValues, nodeId, widgetId, value]);

  useEffect(() => {
    if (streamedValue !== value) {
      onChange(streamedValue);
    }
  }, [streamedValue, value, onChange]);

  return (
    <div style={{ marginBottom: "10px" }}>
      <label style={{ color: "var(--sub-text)", fontSize: "12px" }}>{label}</label>
      <div
        style={{
          backgroundColor: "rgba(0,0,0,0.3)",
          border: "1px solid var(--node-border)",
          borderRadius: "4px",
          color: "var(--primary-color)",
          fontFamily: "monospace",
          fontSize: "13px",
          minHeight: "60px",
          padding: "8px",
          whiteSpace: "pre-wrap",
        }}
      >
        {streamedValue || <span style={{ opacity: 0.3 }}>Waiting for stream...</span>}
      </div>
    </div>
  );
};

const SignalButtonWidget = (props: WidgetProps) => {
  const { id, label } = props;
  const registry = props.registry;
  const formContext = registry.formContext as Record<string, unknown> | undefined;
  const nodeId = formContext?.nodeId as string;
  const widgetId = id.split("_").pop() ?? id;
  const sendWidgetSignal = useFlowStore((s) => s.sendWidgetSignal);

  return (
    <button
      onClick={(e) => {
        e.preventDefault();
        sendWidgetSignal(
          create(WidgetSignalSchema, {
            nodeId: nodeId,
            payload: {
              case: "data",
              value: new Uint8Array([1]), // Signal as binary or structured data instead of JSON
            },
            widgetId: widgetId,
          }),
        );
      }}
      style={{
        backgroundColor: "var(--primary-color)",
        border: "none",
        borderRadius: "4px",
        color: "#fff",
        cursor: "pointer",
        marginTop: "5px",
        padding: "6px",
        width: "100%",
      }}
    >
      {label || "Send Signal"}
    </button>
  );
};

const customWidgets: RegistryWidgetsType = {
  signalButton: SignalButtonWidget,
  streamingText: StreamingTextWidget,
};

const generateUiSchema = (s: RJSFSchema): UiSchema => {
  const ui: UiSchema = {};
  if (s.properties) {
    Object.entries(s.properties).forEach(([key, value]) => {
      const val = value as RJSFSchema;
      // 如果字段定义了 uiWidget，映射到 ui:widget
      if (val.uiWidget) {
        ui[key] = { "ui:widget": val.uiWidget as string };
      }
      // 如果是嵌套对象，递归处理
      if (val.type === "object" && val.properties) {
        const nestedUi = generateUiSchema(val);
        if (Object.keys(nestedUi).length > 0) {
          const existing = (ui[key] as object | undefined) ?? {};
          ui[key] = { ...existing, ...nestedUi };
        }
      }
    });
  }
  ui["ui:submitButtonOptions"] = { norender: true };
  return ui;
};

import { NodeErrorBoundary } from "../base/NodeErrorBoundary";

interface FlowcraftRJSFProps {
  formData: Record<string, unknown>;
  nodeId: string;
  onChange: (data: Record<string, unknown>) => void;
  schema: RJSFSchema;
}

export const FlowcraftRJSF: React.FC<FlowcraftRJSFProps> = React.memo(({ formData, nodeId, onChange, schema }) => {
  const [renderError, setRenderError] = useState<null | string>(null);

  const uiSchema = React.useMemo(() => generateUiSchema(schema), [schema]);

  if (renderError) {
    return (
      <div
        style={{
          background: "rgba(255,77,79,0.1)",
          borderRadius: "4px",
          color: "#ff4d4f",
          fontSize: "12px",
          padding: "10px",
        }}
      >
        <strong>RJSF Render Error:</strong>
        <pre style={{ marginTop: "5px", overflow: "auto" }}>{renderError}</pre>
        <button
          onClick={() => {
            setRenderError(null);
          }}
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <NodeErrorBoundary nodeId={nodeId}>
      <Form
        formContext={{ nodeId }}
        formData={formData}
        onChange={(e) => {
          onChange(e.formData as Record<string, unknown>);
        }}
        schema={schema}
        templates={{ ObjectFieldTemplate: PlainObjectFieldTemplate }}
        uiSchema={uiSchema}
        validator={validator}
        widgets={customWidgets}
      >
        <div />
      </Form>
    </NodeErrorBoundary>
  );
});
