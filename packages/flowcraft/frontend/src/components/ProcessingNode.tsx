import { type NodeProps } from "@xyflow/react";
import { memo } from "react";
import { useTaskStore } from "../store/taskStore";
import { TaskStatus, type ProcessingNodeData } from "../types";
import { useMockSocket } from "../hooks/useMockSocket";
import { BaseNode } from "./base/BaseNode";
import { Handle } from "./base/Handle";
import { Position } from "@xyflow/react";

const ProcessingContent: React.FC<{
  id: string;
  data: ProcessingNodeData;
}> = ({ data }) => {
  const { taskId, label } = data;
  const { cancelTask } = useMockSocket({ disablePolling: true });
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
        height: "100%",
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        boxSizing: "border-box",
      }}
    >
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
            onClick={() => {
              cancelTask(taskId);
            }}
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
    </div>
  );
};

const ProcessingNode: React.FC<NodeProps> = (props) => {
  const { selected, positionAbsoluteX, positionAbsoluteY } = props;

  return (
    <div
      style={{
        borderRadius: "12px",
        background: "var(--node-bg)",
        color: "white",
        border: `2px solid ${selected ? "var(--primary-color)" : "rgba(113, 128, 150, 0.4)"}`,
        minWidth: "200px",
        textAlign: "center",
        boxShadow: selected
          ? "0 0 20px rgba(100, 108, 255, 0.3)"
          : "0 4px 15px rgba(0,0,0,0.3)",
        overflow: "visible", // For floating panel
        transition: "all 0.2s ease",
      }}
    >
      <BaseNode
        {...props}
        renderWidgets={ProcessingContent}
        type="processing"
        x={positionAbsoluteX}
        y={positionAbsoluteY}
        handles={
          <>
            <Handle type="target" position={Position.Top} />
            <Handle type="source" position={Position.Bottom} />
          </>
        }
      />
    </div>
  );
};

export default memo(ProcessingNode);
