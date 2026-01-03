import React, { useEffect, useState } from "react";
import Form from "@rjsf/core";
import validator from "@rjsf/validator-ajv8";
import type { WidgetProps, RegistryWidgetsType } from "@rjsf/utils";
import { useFlowStore } from "../../store/flowStore";
import { socketClient } from "../../utils/SocketClient";
import { WidgetSignalSchema } from "../../generated/flowcraft/v1/core/signals_pb";
import { create } from "@bufbuild/protobuf";

/**
 * 自定义流式文本组件
 * 展示了如何让 RJSF 小部件支持来自后端的流式输出
 */
const StreamingTextWidget = (props: WidgetProps) => {
  const { value, onChange, label, id, registry } = props;
  const nodeId = registry.formContext.nodeId;
  const widgetId = id.split("_").pop(); // RJSF internal ID to simple key

  const [streamedValue, setStreamedValue] = useState(value || "");

  useEffect(() => {
    if (!nodeId || !widgetId) return;

    const handler = (chunk: any) => {
      if (chunk.nodeId === nodeId && chunk.widgetId === widgetId) {
        setStreamedValue((prev: string) => prev + chunk.chunkData);
      }
    };

    socketClient.on("streamChunk", handler);
    return () => {
      socketClient.off("streamChunk", handler);
    };
  }, [nodeId, widgetId]);

  // Sync back to RJSF state when stream changes (optional, depends on sync strategy)
  useEffect(() => {
    if (streamedValue !== value) {
      onChange(streamedValue);
    }
  }, [streamedValue]);

  return (
    <div style={{ marginBottom: "10px" }}>
      <label style={{ fontSize: "12px", color: "var(--sub-text)" }}>
        {label}
      </label>
      <div
        style={{
          minHeight: "60px",
          padding: "8px",
          backgroundColor: "rgba(0,0,0,0.3)",
          border: "1px solid var(--node-border)",
          borderRadius: "4px",
          fontFamily: "monospace",
          fontSize: "13px",
          whiteSpace: "pre-wrap",
          color: "var(--primary-color)",
        }}
      >
        {streamedValue || (
          <span style={{ opacity: 0.3 }}>Waiting for stream...</span>
        )}
      </div>
    </div>
  );
};

/**
 * 自定义信号按钮
 * 展示了如何让 RJSF 小部件支持双向通信
 */
const SignalButtonWidget = (props: WidgetProps) => {
  const { label, id, registry } = props;
  const nodeId = registry.formContext.nodeId;
  const widgetId = id.split("_").pop() || id;
  const sendWidgetSignal = useFlowStore((s) => s.sendWidgetSignal);

  return (
    <button
      onClick={(e) => {
        e.preventDefault();
        sendWidgetSignal(
          create(WidgetSignalSchema, {
            nodeId: nodeId,
            widgetId: widgetId,
            dataJson: JSON.stringify({ action: "trigger" }),
          }),
        );
      }}
      style={{
        width: "100%",
        padding: "6px",
        backgroundColor: "var(--primary-color)",
        color: "#fff",
        border: "none",
        borderRadius: "4px",
        cursor: "pointer",
        marginTop: "5px",
      }}
    >
      {label || "Send Signal"}
    </button>
  );
};

const customWidgets: RegistryWidgetsType = {
  streamingText: StreamingTextWidget,
  signalButton: SignalButtonWidget,
};

interface FlowcraftRJSFProps {
  schema: any;
  formData: any;
  onChange: (data: any) => void;
  nodeId: string;
}

export const FlowcraftRJSF: React.FC<FlowcraftRJSFProps> = ({
  schema,
  formData,
  onChange,
  nodeId,
}) => {
  // Generate UI Schema automatically based on uiWidget hint in schema
  const generateUiSchema = (s: any) => {
    const ui: any = {};
    if (s.properties) {
      Object.entries(s.properties).forEach(([key, value]: [string, any]) => {
        if (value.uiWidget) {
          ui[key] = { "ui:widget": value.uiWidget };
        }
      });
    }
    ui["ui:submitButtonOptions"] = { norender: true };
    return ui;
  };

  const uiSchema = React.useMemo(() => generateUiSchema(schema), [schema]);

  return (
    <Form
      schema={schema}
      formData={formData}
      uiSchema={uiSchema}
      validator={validator}
      widgets={customWidgets}
      formContext={{ nodeId }}
      onChange={(e) => {
        onChange(e.formData);
      }}
    >
      <div />
    </Form>
  );
};
