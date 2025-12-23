import { useState } from "react";
import { useNotificationStore } from "../store/notificationStore";

export function Notifications() {
  const { notifications } = useNotificationStore();
  const [showHistory, setShowHistory] = useState(false);

  return (
    <div
      style={{
        position: "absolute",
        top: 10,
        right: 10,
        zIndex: 1000,
        display: "flex",
        flexDirection: "column",
        alignItems: "flex-end",
      }}
    >
      <button onClick={() => setShowHistory(!showHistory)}>
        {showHistory ? "Hide" : "Show"} Notification History
      </button>
      {showHistory && (
        <div
          style={{
            marginTop: 10,
            padding: 10,
            backgroundColor: "white",
            border: "1px solid black",
            maxHeight: "300px",
            overflowY: "auto",
          }}
        >
          <h3>Notification History</h3>
          <ul>
            {notifications.map((n) => (
              <li key={n.id}>
                <strong>{n.type.toUpperCase()}:</strong> {n.message} (
                {n.timestamp.toLocaleTimeString()})
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
