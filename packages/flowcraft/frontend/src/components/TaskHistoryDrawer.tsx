import React, { useMemo } from "react";

import { useTaskStore } from "../store/taskStore";
import { MutationSource, TaskStatus } from "../types";

export const TaskHistoryDrawer: React.FC = () => {
  const {
    isDrawerOpen,
    mutationLogs,
    selectedTaskId,
    setDrawerOpen,
    setSelectedTaskId,
    tasks,
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
          alignItems: "center",
          background: "var(--node-bg)",
          border: "1px solid var(--node-border)",
          borderBottom: "none",
          borderRadius: "8px 8px 0 0",
          bottom: 0,
          cursor: "pointer",
          display: "flex",
          fontSize: "12px",
          gap: "8px",
          padding: "8px 16px",
          position: "fixed",
          right: "20px",
          zIndex: 1000,
        }}
      >
        <span>📊 Task History</span>
        <span
          style={{
            background: "var(--primary-color)",
            borderRadius: "10px",
            fontSize: "10px",
            padding: "2px 6px",
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
        background: "#1a1a1a",
        borderTop: "1px solid #333",
        bottom: 0,
        color: "#ccc",
        display: "flex",
        flexDirection: "column",
        fontFamily: "monospace",
        height: "300px",
        left: 0,
        position: "fixed",
        right: 0,
        zIndex: 2000,
      }}
    >
      {/* Header */}
      <div
        style={{
          alignItems: "center",
          background: "#222",
          borderBottom: "1px solid #333",
          display: "flex",
          justifyContent: "space-between",
          padding: "8px 16px",
        }}
      >
        <div style={{ alignItems: "center", display: "flex", gap: "20px" }}>
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
            borderRight: "1px solid #333",
            overflowY: "auto",
            width: "300px",
          }}
        >
          {taskList.map((task) => (
            <div
              key={task.taskId}
              onClick={() => {
                setSelectedTaskId(task.taskId);
              }}
              style={{
                background:
                  selectedTaskId === task.taskId ? "#2d2d2d" : "transparent",
                borderBottom: "1px solid #222",
                cursor: "pointer",
                fontSize: "12px",
                padding: "8px 12px",
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
                      task.source === MutationSource.SOURCE_USER
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
              style={{ marginTop: "40px", opacity: 0.5, textAlign: "center" }}
            >
              Select a task to view mutation details
            </div>
          ) : (
            <div>
              <div
                style={{
                  borderBottom: "1px solid #333",
                  marginBottom: "16px",
                  paddingBottom: "8px",
                }}
              >
                <h3 style={{ color: "#fff", margin: 0 }}>
                  {selectedTask.label}
                </h3>
                <div
                  style={{ fontSize: "11px", marginTop: "4px", opacity: 0.7 }}
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
                    style={{ fontSize: "12px", marginBottom: "12px" }}
                  >
                    <div style={{ color: "#ed8936", marginBottom: "4px" }}>
                      ➜ {log.description}
                    </div>
                    <div
                      style={{
                        borderLeft: "2px solid #333",
                        opacity: 0.8,
                        paddingLeft: "12px",
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
