import { useState } from "react";

import { useTheme } from "@/hooks/useTheme";
import { useNotificationStore } from "@/store/notificationStore";

export function Notifications() {
  const { clearNotifications, notifications } = useNotificationStore();
  const [isOpen, setIsOpen] = useState(false);
  const { theme } = useTheme();

  const isDark = theme === "dark";

  const buttonStyle: React.CSSProperties = {
    alignItems: "center",
    backdropFilter: "blur(4px)",
    backgroundColor: isDark ? "rgba(0, 0, 0, 0.5)" : "rgba(255, 255, 255, 0.8)",
    border: `1px solid ${isDark ? "rgba(255, 255, 255, 0.1)" : "rgba(0, 0, 0, 0.1)"}`,
    borderRadius: "20px",
    boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
    color: isDark ? "rgba(240, 240, 240, 0.8)" : "rgba(33, 53, 71, 0.8)",
    cursor: "pointer",
    display: "flex",
    fontSize: "12px",
    gap: "6px",
    padding: "6px 12px",
    position: "absolute",
    right: 10,
    top: 10,
    transition: "all 0.2s ease",
    zIndex: 1000,
  };

  const drawerStyle: React.CSSProperties = {
    backgroundColor: isDark ? "#1a1a1a" : "#ffffff",
    borderLeft: `1px solid ${isDark ? "#333" : "#eee"}`,
    boxShadow: isOpen ? "-2px 0 8px rgba(0,0,0,0.2)" : "none",
    display: "flex",
    flexDirection: "column",
    height: "100vh",
    position: "fixed",
    right: isOpen ? 0 : "-320px",
    top: 0,
    transition: "right 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
    width: "300px",
    zIndex: 1001,
  };

  const headerStyle: React.CSSProperties = {
    alignItems: "center",
    borderBottom: `1px solid ${isDark ? "#333" : "#eee"}`,
    display: "flex",
    justifyContent: "space-between",
    padding: "16px",
  };

  const titleStyle: React.CSSProperties = {
    color: isDark ? "#f0f0f0" : "#333",
    fontSize: "16px",
    fontWeight: 600,
    margin: 0,
  };

  const listStyle: React.CSSProperties = {
    flex: 1,
    listStyle: "none",
    margin: "0",
    overflowY: "auto",
    padding: "0",
  };

  const itemStyle: React.CSSProperties = {
    borderBottom: `1px solid ${isDark ? "#333" : "#f5f5f5"}`,
    color: isDark ? "#ccc" : "#555",
    fontSize: "13px",
    padding: "12px 16px",
  };

  const backdropStyle: React.CSSProperties = {
    backgroundColor: "rgba(0,0,0,0.3)",
    height: "100vh",
    left: 0,
    opacity: isOpen ? 1 : 0,
    pointerEvents: isOpen ? "auto" : "none",
    position: "fixed",
    top: 0,
    transition: "opacity 0.3s ease",
    width: "100vw",
    zIndex: 1000,
  };

  return (
    <>
      <button
        onClick={() => {
          setIsOpen(true);
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.backgroundColor = isDark
            ? "rgba(255, 255, 255, 0.1)"
            : "rgba(0, 0, 0, 0.05)";
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.backgroundColor = isDark
            ? "rgba(0, 0, 0, 0.5)"
            : "rgba(255, 255, 255, 0.8)";
        }}
        style={buttonStyle}
      >
        <span>🔔</span>
        <span>History</span>
        {notifications.length > 0 && (
          <span
            style={{
              alignItems: "center",
              backgroundColor: "#ff4d4f",
              borderRadius: "10px",
              color: "white",
              display: "flex",
              fontSize: "10px",
              height: "16px",
              justifyContent: "center",
              padding: "0 6px",
            }}
          >
            {notifications.length}
          </span>
        )}
      </button>

      <div
        onClick={() => {
          setIsOpen(false);
        }}
        style={backdropStyle}
      />

      <div style={drawerStyle}>
        <div style={headerStyle}>
          <h3 style={titleStyle}>Notifications</h3>
          <div style={{ display: "flex", gap: "8px" }}>
            {notifications.length > 0 && (
              <button
                onClick={clearNotifications}
                style={{
                  background: "none",
                  border: "none",
                  color: "#ff4d4f",
                  cursor: "pointer",
                  fontSize: "12px",
                  padding: "4px 8px",
                }}
              >
                Clear
              </button>
            )}
            <button
              onClick={() => {
                setIsOpen(false);
              }}
              style={{
                background: "none",
                border: "none",
                color: isDark ? "#999" : "#666",
                cursor: "pointer",
                fontSize: "16px",
                lineHeight: 1,
                padding: "4px",
              }}
            >
              ✕
            </button>
          </div>
        </div>

        <ul style={listStyle}>
          {notifications.length === 0 ? (
            <li
              style={{
                color: isDark ? "#666" : "#999",
                fontSize: "13px",
                padding: "20px",
                textAlign: "center",
              }}
            >
              No notifications yet
            </li>
          ) : (
            notifications
              .slice()
              .reverse()
              .map((n) => (
                <li key={n.id} style={itemStyle}>
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      marginBottom: "4px",
                    }}
                  >
                    <strong
                      style={{
                        color:
                          n.type === "error"
                            ? "#ff4d4f"
                            : n.type === "success"
                              ? "#52c41a"
                              : isDark
                                ? "#1890ff"
                                : "#096dd9",
                        fontSize: "11px",
                        letterSpacing: "0.5px",
                        textTransform: "uppercase",
                      }}
                    >
                      {n.type}
                    </strong>
                    <span
                      style={{
                        color: isDark ? "#666" : "#999",
                        fontSize: "11px",
                      }}
                    >
                      {n.timestamp.toLocaleTimeString([], {
                        hour: "2-digit",
                        minute: "2-digit",
                      })}
                    </span>
                  </div>
                  <div style={{ lineHeight: "1.4" }}>{n.message}</div>
                </li>
              ))
          )}
        </ul>
      </div>
    </>
  );
}
