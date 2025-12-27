import React, { useMemo } from "react";
import { useTaskStore } from "../store/taskStore";
import { MutationSource, TaskStatus } from "../types";

export const TaskHistoryDrawer: React.FC = () => {
  const {
    tasks,
    mutationLogs,
    isDrawerOpen,
    setDrawerOpen,
    selectedTaskId,
    setSelectedTaskId,
  } = useTaskStore();

  const taskList = useMemo(
    () => Object.values(tasks).sort((a, b) => b.createdAt - a.createdAt),
    [tasks],
  );

  if (!isDrawerOpen) {
    return (
      <div
        onClick={() => {
          setDrawerOpen(true);
        }}
        style={{
          position: "fixed",
          bottom: 0,
          right: "20px",
          background: "var(--node-bg)",
          padding: "8px 16px",
          border: "1px solid var(--node-border)",
          borderBottom: "none",
          borderRadius: "8px 8px 0 0",
          cursor: "pointer",
          zIndex: 1000,
          fontSize: "12px",
          display: "flex",
          alignItems: "center",
          gap: "8px",
        }}
      >
        <span>📊 Task History</span>
        <span
          style={{
            background: "var(--primary-color)",
            padding: "2px 6px",
            borderRadius: "10px",
            fontSize: "10px",
          }}
        >
          {taskList.length}
        </span>
      </div>
    );
  }

  const selectedTask = selectedTaskId ? tasks[selectedTaskId] : null;
  const relatedLogs = mutationLogs.filter(
    (log) => log.taskId === selectedTaskId,
  );

  return (
    <div
      style={{
        position: "fixed",
        bottom: 0,
        left: 0,
        right: 0,
        height: "300px",
        background: "#1a1a1a",
        borderTop: "1px solid #333",
        zIndex: 2000,
        display: "flex",
        flexDirection: "column",
        color: "#ccc",
        fontFamily: "monospace",
      }}
    >
      {/* Header */}
      <div
        style={{
          padding: "8px 16px",
          background: "#222",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          borderBottom: "1px solid #333",
        }}
      >
        <div style={{ display: "flex", gap: "20px", alignItems: "center" }}>
          <b style={{ color: "#fff" }}>ACTIVITY TRACER</b>
          <div style={{ fontSize: "11px" }}>
            Total Tasks: {taskList.length} | Logs: {mutationLogs.length}
          </div>
        </div>
        <button
          onClick={() => {
            setDrawerOpen(false);
          }}
          style={{
            background: "transparent",
            border: "none",
            color: "#666",
            cursor: "pointer",
          }}
        >
          [Minimize]
        </button>
      </div>

      <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>
        {/* Task List */}
        <div
          style={{
            width: "300px",
            borderRight: "1px solid #333",
            overflowY: "auto",
          }}
        >
          {taskList.map((task) => (
            <div
              key={task.taskId}
              onClick={() => {
                setSelectedTaskId(task.taskId);
              }}
              style={{
                padding: "8px 12px",
                borderBottom: "1px solid #222",
                cursor: "pointer",
                background:
                  selectedTaskId === task.taskId ? "#2d2d2d" : "transparent",
                fontSize: "12px",
              }}
            >
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  marginBottom: "4px",
                }}
              >
                <span
                  style={{
                    color:
                      task.source === MutationSource.USER
                        ? "#4299e1"
                        : "#48bb78",
                    fontWeight: "bold",
                  }}
                >
                  [{task.source}]
                </span>
                <span style={{ fontSize: "10px", opacity: 0.5 }}>
                  {new Date(task.createdAt).toLocaleTimeString()}
                </span>
              </div>
              <div
                style={{
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                }}
              >
                {task.label}
              </div>
            </div>
          ))}
        </div>

        {/* Mutation Log Detail */}
        <div style={{ flex: 1, overflowY: "auto", padding: "16px" }}>
          {!selectedTask ? (
            <div
              style={{ opacity: 0.5, textAlign: "center", marginTop: "40px" }}
            >
              Select a task to view mutation details
            </div>
          ) : (
            <div>
              <div
                style={{
                  marginBottom: "16px",
                  borderBottom: "1px solid #333",
                  paddingBottom: "8px",
                }}
              >
                <h3 style={{ margin: 0, color: "#fff" }}>
                  {selectedTask.label}
                </h3>
                <div
                  style={{ fontSize: "11px", opacity: 0.7, marginTop: "4px" }}
                >
                  ID: {selectedTask.taskId} | Status:{" "}
                  {Object.keys(TaskStatus).find(
                    (k) =>
                      TaskStatus[k as keyof typeof TaskStatus] ===
                      selectedTask.status,
                  )}
                </div>
              </div>

              {relatedLogs.length === 0 ? (
                <div style={{ opacity: 0.5 }}>
                  No mutations recorded for this task.
                </div>
              ) : (
                relatedLogs.map((log) => (
                  <div
                    key={log.id}
                    style={{ marginBottom: "12px", fontSize: "12px" }}
                  >
                    <div style={{ color: "#ed8936", marginBottom: "4px" }}>
                      ➜ {log.description}
                    </div>
                    <div
                      style={{
                        paddingLeft: "12px",
                        borderLeft: "2px solid #333",
                        opacity: 0.8,
                      }}
                    >
                      {log.mutations.map((m, idx) => {
                        const type = Object.keys(m).find((k) => k !== "toJSON");
                        return (
                          <div key={idx} style={{ marginBottom: "2px" }}>
                            • {type}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                ))
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
