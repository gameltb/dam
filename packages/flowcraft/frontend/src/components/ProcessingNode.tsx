import { type NodeProps } from "@xyflow/react";
import { Position } from "@xyflow/react";
import { memo } from "react";

import { useFlowSocket } from "../hooks/useFlowSocket";
import { useTaskStore } from "../store/taskStore";
import {
  type ProcessingNodeData,
  type ProcessingNodeType,
  TaskStatus,
} from "../types";
import { BaseNode } from "./base/BaseNode";
import { Handle } from "./base/Handle";
import { NodeErrorBoundary } from "./base/NodeErrorBoundary";

const ProcessingContent: React.FC<{
  data: ProcessingNodeData;
  id: string;
}> = ({ data }) => {
  const { label, taskId } = data;
  const { cancelTask } = useFlowSocket({ disablePolling: true });
  const taskState = useTaskStore((state) => state.tasks[taskId]);

  const progress = taskState?.progress ?? 0;
  const status = taskState?.status ?? TaskStatus.TASK_PENDING;
  const message = taskState?.message ?? "Initializing...";

  const getStatusLabel = (s: TaskStatus) => {
    switch (s) {
      case TaskStatus.TASK_CANCELLED:
        return "CANCELLED";
      case TaskStatus.TASK_COMPLETED:
        return "COMPLETED";
      case TaskStatus.TASK_FAILED:
        return "FAILED";
      case TaskStatus.TASK_PENDING:
        return "PENDING";
      case TaskStatus.TASK_PROCESSING:
        return "PROCESSING";
      default:
        return "UNKNOWN";
    }
  };

  return (
    <div
      style={{
        boxSizing: "border-box",
        display: "flex",
        flexDirection: "column",
        height: "100%",
        justifyContent: "center",
        padding: "16px",
      }}
    >
      <div style={{ fontWeight: "bold", marginBottom: "8px" }}>{label}</div>

      <div style={{ color: "#cbd5e0", fontSize: "12px", marginBottom: "4px" }}>
        {getStatusLabel(status)}
      </div>

      <div
        style={{
          background: "#4a5568",
          borderRadius: "3px",
          height: "6px",
          marginBottom: "8px",
          overflow: "hidden",
          width: "100%",
        }}
      >
        <div
          style={{
            background:
              status === TaskStatus.TASK_FAILED ? "#e53e3e" : "#4299e1",
            height: "100%",
            transition: "width 0.3s ease",
            width: `${String(Math.round(progress))}%`,
          }}
        />
      </div>

      <div style={{ color: "#a0aec0", fontSize: "10px", marginBottom: "12px" }}>
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
              borderRadius: "4px",
              color: "#e53e3e",
              cursor: "pointer",
              fontSize: "10px",
              padding: "4px 8px",
            }}
          >
            Cancel
          </button>
        )}
    </div>
  );
};

const ProcessingNode: React.FC<NodeProps<ProcessingNodeType>> = (props) => {
  const { id, positionAbsoluteX, positionAbsoluteY, selected } = props;

  return (
    <div
      className={selected ? "fc-node fc-node-selected" : "fc-node"}
      style={{
        minWidth: "200px",
        overflow: "visible", // For floating panel
        textAlign: "center",
      }}
    >
      <NodeErrorBoundary nodeId={id}>
        <BaseNode<ProcessingNodeType>
          {...props}
          handles={
            <>
              <Handle position={Position.Top} type="target" />
              <Handle position={Position.Bottom} type="source" />
            </>
          }
          renderWidgets={ProcessingContent}
          type="processing"
          x={positionAbsoluteX}
          y={positionAbsoluteY}
        />
      </NodeErrorBoundary>
    </div>
  );
};

export default memo(ProcessingNode);
