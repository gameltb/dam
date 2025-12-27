import React, { useState } from "react";
import { useTheme } from "../hooks/useTheme";

interface EditUrlModalProps {
  currentUrl: string;
  onClose: () => void;
  onSave: (newUrl: string) => void;
}

export const EditUrlModal = ({
  currentUrl,
  onClose,
  onSave,
}: EditUrlModalProps) => {
  const { theme } = useTheme();
  const [url, setUrl] = useState(currentUrl);

  const handleSave = () => {
    onSave(url);
    onClose();
  };

  const modalOverlayStyle: React.CSSProperties = {
    position: "fixed",
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: "rgba(0, 0, 0, 0.5)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    zIndex: 1000,
  };

  const modalContentStyle: React.CSSProperties = {
    backgroundColor: theme === "dark" ? "#2a2a2a" : "white",
    padding: 20,
    borderRadius: 5,
    width: 400,
    color: theme === "dark" ? "#f0f0f0" : "#213547",
  };

  const inputStyle: React.CSSProperties = {
    width: "100%",
    padding: "8px",
    marginBottom: "10px",
    boxSizing: "border-box",
  };

  return (
    <div style={modalOverlayStyle}>
      <div style={modalContentStyle}>
        <h2>Edit WebSocket URL</h2>
        <input
          type="text"
          value={url}
          onChange={(e) => {
            setUrl(e.target.value);
          }}
          style={inputStyle}
        />
        <div
          style={{ display: "flex", justifyContent: "flex-end", marginTop: 10 }}
        >
          <button onClick={onClose} style={{ marginRight: 10 }}>
            Close
          </button>
          <button onClick={handleSave}>Save & Connect</button>
        </div>
      </div>
    </div>
  );
};
