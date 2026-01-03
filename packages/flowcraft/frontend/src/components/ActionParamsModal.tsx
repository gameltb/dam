import React from "react";
import Form from "@rjsf/core";
import type { IChangeEvent } from "@rjsf/core";
import validator from "@rjsf/validator-ajv8";
import type { ActionTemplate } from "../generated/flowcraft/v1/core/action_pb";
import { getSchemaForTemplate } from "../utils/schemaRegistry";

interface ActionParamsModalProps {
  action: ActionTemplate;
  onConfirm: (params: Record<string, any>) => void;
  onCancel: () => void;
}

export const ActionParamsModal: React.FC<ActionParamsModalProps> = ({
  action,
  onConfirm,
  onCancel,
}) => {
  const schema = React.useMemo(() => {
    return getSchemaForTemplate(action.id, action.paramsSchemaJson);
  }, [action]);

  if (!schema) {
    // If no schema found, execute immediately
    React.useEffect(() => {
      onConfirm({});
    }, []);
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
      onConfirm(data.formData);
    }
  };

  return (
    <div
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: "rgba(0,0,0,0.6)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 10000,
        backdropFilter: "blur(4px)",
      }}
    >
      <div
        style={{
          backgroundColor: "var(--panel-bg)",
          padding: "24px",
          borderRadius: "12px",
          width: "500px",
          maxHeight: "80vh",
          overflowY: "auto",
          border: "1px solid var(--node-border)",
          boxShadow: "0 20px 40px rgba(0,0,0,0.4)",
          color: "var(--text-color)",
        }}
      >
        <h3 style={{ marginTop: 0, color: "var(--primary-color)" }}>
          {action.label}
        </h3>

        <div className="rjsf-container" style={{ margin: "20px 0" }}>
          <Form
            schema={schema}
            uiSchema={uiSchema}
            validator={validator}
            onSubmit={handleSubmit}
          >
            <div
              style={{
                display: "flex",
                gap: "12px",
                justifyContent: "flex-end",
                marginTop: "20px",
              }}
            >
              <button
                type="button"
                onClick={onCancel}
                style={{
                  padding: "8px 16px",
                  borderRadius: "6px",
                  background: "none",
                  border: "1px solid var(--node-border)",
                  color: "var(--text-color)",
                  cursor: "pointer",
                }}
              >
                Cancel
              </button>
              <button
                type="submit"
                style={{
                  padding: "8px 16px",
                  borderRadius: "6px",
                  background: "var(--primary-color)",
                  border: "none",
                  color: "#fff",
                  cursor: "pointer",
                  fontWeight: 600,
                }}
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
