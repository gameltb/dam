import type { IChangeEvent } from "@rjsf/core";

import Form from "@rjsf/core";
import validator from "@rjsf/validator-ajv8";
import React from "react";

import type { ActionTemplate } from "@/generated/flowcraft/v1/core/action_pb";

import { getSchemaForTemplate } from "@/utils/schemaRegistry";

interface ActionParamsModalProps {
  action: ActionTemplate;
  onCancel: () => void;
  onConfirm: (params: Record<string, unknown>) => void;
}

export const ActionParamsModal: React.FC<ActionParamsModalProps> = ({ action, onCancel, onConfirm }) => {
  const schema = React.useMemo(() => {
    return getSchemaForTemplate(action.id);
  }, [action]);

  React.useEffect(() => {
    if (!schema) {
      onConfirm({});
    }
  }, [schema, onConfirm]);

  if (!schema) {
    return null;
  }

  // Define UI Schema for custom styling
  const uiSchema = {
    "ui:submitButtonOptions": {
      norender: true, // We use our own footer buttons
    },
  };

  const handleSubmit = (data: IChangeEvent) => {
    if (data.formData) {
      onConfirm(data.formData as Record<string, unknown>);
    }
  };

  return (
    <div
      style={{
        alignItems: "center",
        backdropFilter: "blur(4px)",
        backgroundColor: "rgba(0,0,0,0.6)",
        bottom: 0,
        display: "flex",
        justifyContent: "center",
        left: 0,
        position: "fixed",
        right: 0,
        top: 0,
        zIndex: 10000,
      }}
    >
      <div
        style={{
          backgroundColor: "var(--panel-bg)",
          border: "1px solid var(--node-border)",
          borderRadius: "12px",
          boxShadow: "0 20px 40px rgba(0,0,0,0.4)",
          color: "var(--text-color)",
          maxHeight: "80vh",
          overflowY: "auto",
          padding: "24px",
          width: "500px",
        }}
      >
        <h3 style={{ color: "var(--primary-color)", marginTop: 0 }}>{action.label}</h3>

        <div className="rjsf-container" style={{ margin: "20px 0" }}>
          <Form onSubmit={handleSubmit} schema={schema} uiSchema={uiSchema} validator={validator}>
            <div
              style={{
                display: "flex",
                gap: "12px",
                justifyContent: "flex-end",
                marginTop: "20px",
              }}
            >
              <button
                onClick={onCancel}
                style={{
                  background: "none",
                  border: "1px solid var(--node-border)",
                  borderRadius: "6px",
                  color: "var(--text-color)",
                  cursor: "pointer",
                  padding: "8px 16px",
                }}
                type="button"
              >
                Cancel
              </button>
              <button
                style={{
                  background: "var(--primary-color)",
                  border: "none",
                  borderRadius: "6px",
                  color: "#fff",
                  cursor: "pointer",
                  fontWeight: 600,
                  padding: "8px 16px",
                }}
                type="submit"
              >
                Execute Action
              </button>
            </div>
          </Form>
        </div>
      </div>
    </div>
  );
};
