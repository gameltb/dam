import { useState } from "react";
import { useNotificationStore } from "../store/notificationStore";
import { useTheme } from "../hooks/useTheme";

export function Notifications() {
  const { notifications, clearNotifications } = useNotificationStore();
  const [isOpen, setIsOpen] = useState(false);
  const { theme } = useTheme();

  const isDark = theme === "dark";

  const buttonStyle: React.CSSProperties = {
    position: "absolute",
    top: 10,
    right: 10,
    zIndex: 1000,
    cursor: "pointer",
    fontSize: "12px",
    color: isDark ? "rgba(240, 240, 240, 0.8)" : "rgba(33, 53, 71, 0.8)",
    backgroundColor: isDark ? "rgba(0, 0, 0, 0.5)" : "rgba(255, 255, 255, 0.8)",
    padding: "6px 12px",
    border: `1px solid ${isDark ? "rgba(255, 255, 255, 0.1)" : "rgba(0, 0, 0, 0.1)"}`,
    borderRadius: "20px",
    display: "flex",
    alignItems: "center",
    gap: "6px",
    backdropFilter: "blur(4px)",
    boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
    transition: "all 0.2s ease",
  };

  const drawerStyle: React.CSSProperties = {
    position: "fixed",
    top: 0,
    right: isOpen ? 0 : "-320px",
    width: "300px",
    height: "100vh",
    backgroundColor: isDark ? "#1a1a1a" : "#ffffff",
    boxShadow: isOpen ? "-2px 0 8px rgba(0,0,0,0.2)" : "none",
    zIndex: 1001,
    transition: "right 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
    display: "flex",
    flexDirection: "column",
    borderLeft: `1px solid ${isDark ? "#333" : "#eee"}`,
  };

  const headerStyle: React.CSSProperties = {
    padding: "16px",
    borderBottom: `1px solid ${isDark ? "#333" : "#eee"}`,
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
  };

  const titleStyle: React.CSSProperties = {
    margin: 0,
    fontSize: "16px",
    fontWeight: 600,
    color: isDark ? "#f0f0f0" : "#333",
  };

  const listStyle: React.CSSProperties = {
    flex: 1,
    overflowY: "auto",
    padding: "0",
    margin: "0",
    listStyle: "none",
  };

  const itemStyle: React.CSSProperties = {
    padding: "12px 16px",
    borderBottom: `1px solid ${isDark ? "#333" : "#f5f5f5"}`,
    fontSize: "13px",
    color: isDark ? "#ccc" : "#555",
  };

  const backdropStyle: React.CSSProperties = {
    position: "fixed",
    top: 0,
    left: 0,
    width: "100vw",
    height: "100vh",
    backgroundColor: "rgba(0,0,0,0.3)",
    zIndex: 1000,
    opacity: isOpen ? 1 : 0,
    pointerEvents: isOpen ? "auto" : "none",
    transition: "opacity 0.3s ease",
  };

  return (
    <>
      <button
        style={buttonStyle}
        onClick={() => setIsOpen(true)}
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
      >
        <span>🔔</span>
        <span>History</span>
        {notifications.length > 0 && (
          <span
            style={{
              backgroundColor: "#ff4d4f",
              color: "white",
              borderRadius: "10px",
              padding: "0 6px",
              fontSize: "10px",
              height: "16px",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            {notifications.length}
          </span>
        )}
      </button>

      <div style={backdropStyle} onClick={() => setIsOpen(false)} />

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
              onClick={() => setIsOpen(false)}
              style={{
                background: "none",
                border: "none",
                color: isDark ? "#999" : "#666",
                cursor: "pointer",
                fontSize: "16px",
                padding: "4px",
                lineHeight: 1,
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
                padding: "20px",
                textAlign: "center",
                color: isDark ? "#666" : "#999",
                fontSize: "13px",
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
                        textTransform: "uppercase",
                        letterSpacing: "0.5px",
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
