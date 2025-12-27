import React, { memo } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";
import { useTaskStore } from "../store/taskStore";
import { TaskStatus, type ProcessingNodeData } from "../types";

const ProcessingNode: React.FC<NodeProps> = ({ data }) => {
  const { taskId, label, onCancel } = data as ProcessingNodeData;
  const taskState = useTaskStore((state) => state.tasks[taskId]);

  const progress = taskState?.progress ?? 0;
  const status = taskState?.status ?? TaskStatus.TASK_PENDING;
  const message = taskState?.message ?? "Initializing...";

  const getStatusLabel = (s: TaskStatus) => {
    switch (s) {
      case TaskStatus.TASK_PENDING:
        return "PENDING";
      case TaskStatus.TASK_PROCESSING:
        return "PROCESSING";
      case TaskStatus.TASK_COMPLETED:
        return "COMPLETED";
      case TaskStatus.TASK_FAILED:
        return "FAILED";
      case TaskStatus.TASK_CANCELLED:
        return "CANCELLED";
      default:
        return "UNKNOWN";
    }
  };

  return (
    <div
      style={{
        padding: "16px",
        borderRadius: "8px",
        background: "#2d3748",
        color: "white",
        border: "1px dashed #718096",
        minWidth: "200px",
        textAlign: "center",
        boxShadow: "0 4px 6px rgba(0,0,0,0.1)",
      }}
    >
      <Handle
        type="target"
        position={Position.Top}
        style={{ background: "#555" }}
      />

      <div style={{ fontWeight: "bold", marginBottom: "8px" }}>{label}</div>

      <div style={{ fontSize: "12px", color: "#cbd5e0", marginBottom: "4px" }}>
        {getStatusLabel(status)}
      </div>

      <div
        style={{
          width: "100%",
          height: "6px",
          background: "#4a5568",
          borderRadius: "3px",
          overflow: "hidden",
          marginBottom: "8px",
        }}
      >
        <div
          style={{
            width: `${String(Math.round(progress))}%`,
            height: "100%",
            background:
              status === TaskStatus.TASK_FAILED ? "#e53e3e" : "#4299e1",
            transition: "width 0.3s ease",
          }}
        />
      </div>

      <div style={{ fontSize: "10px", color: "#a0aec0", marginBottom: "12px" }}>
        {message}
      </div>

      {status !== TaskStatus.TASK_COMPLETED &&
        status !== TaskStatus.TASK_CANCELLED &&
        status !== TaskStatus.TASK_FAILED && (
          <button
            className="nodrag"
            onClick={() => onCancel?.(taskId)}
            style={{
              background: "transparent",
              border: "1px solid #e53e3e",
              color: "#e53e3e",
              borderRadius: "4px",
              padding: "4px 8px",
              fontSize: "10px",
              cursor: "pointer",
            }}
          >
            Cancel
          </button>
        )}

      <Handle
        type="source"
        position={Position.Bottom}
        style={{ background: "#555" }}
      />
    </div>
  );
};

export default memo(ProcessingNode);
