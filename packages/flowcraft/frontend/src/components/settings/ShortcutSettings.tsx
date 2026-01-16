import React, { useEffect, useState } from "react";

import { type ShortcutConfig } from "@/store/uiStore";

interface ShortcutSettingsProps {
  setShortcut: (key: keyof ShortcutConfig, val: string) => void;
  shortcuts: ShortcutConfig;
}

const ShortcutRecordButton: React.FC<{
  label: string;
  onSave: (val: string) => void;
  value: string;
}> = ({ onSave, value }) => {
  const [isRecording, setIsRecording] = useState(false);

  useEffect(() => {
    if (!isRecording) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      e.preventDefault();
      e.stopPropagation();

      const keys = [];
      if (e.ctrlKey || e.metaKey) keys.push("mod");
      if (e.shiftKey) keys.push("shift");
      if (e.altKey) keys.push("alt");

      const key = e.key.toLowerCase();
      if (key !== "control" && key !== "shift" && key !== "alt" && key !== "meta") {
        keys.push(key);
        onSave(keys.join("+"));
        setIsRecording(false);
      }
    };

    window.addEventListener("keydown", handleKeyDown, true);
    return () => {
      window.removeEventListener("keydown", handleKeyDown, true);
    };
  }, [isRecording, onSave]);

  return (
    <button
      onClick={() => {
        setIsRecording(true);
      }}
      style={{
        backgroundColor: isRecording ? "rgba(100, 108, 255, 0.2)" : "rgba(255,255,255,0.08)",
        border: `1px solid ${isRecording ? "var(--primary-color)" : "rgba(255,255,255,0.1)"}`,
        borderRadius: "6px",
        color: isRecording ? "var(--primary-color)" : "#fff",
        cursor: "pointer",
        fontFamily: "monospace",
        fontSize: "11px",
        minWidth: "80px",
        padding: "4px 10px",
        textAlign: "center",
      }}
    >
      {isRecording ? "Press keys..." : value}
    </button>
  );
};

export const ShortcutSettings: React.FC<ShortcutSettingsProps> = ({ setShortcut, shortcuts }) => {
  const shortcutList: { key: keyof ShortcutConfig; label: string }[] = [
    { key: "undo", label: "Undo" },
    { key: "redo", label: "Redo" },
    { key: "copy", label: "Copy" },
    { key: "paste", label: "Paste" },
    { key: "duplicate", label: "Duplicate" },
    { key: "delete", label: "Delete" },
    { key: "autoLayout", label: "Auto Layout" },
  ];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
      {shortcutList.map((s, i) => (
        <div
          key={i}
          style={{
            alignItems: "center",
            borderBottom: "1px solid rgba(255,255,255,0.05)",
            display: "flex",
            justifyContent: "space-between",
            padding: "8px 0",
          }}
        >
          <span style={{ fontSize: "13px" }}>{s.label}</span>
          <ShortcutRecordButton
            label={s.label}
            onSave={(val) => {
              setShortcut(s.key, val);
            }}
            value={shortcuts[s.key]}
          />
        </div>
      ))}

      <div
        style={{
          backgroundColor: "rgba(100, 108, 255, 0.05)",
          borderRadius: "6px",
          color: "var(--sub-text)",
          fontSize: "11px",
          fontStyle: "italic",
          marginTop: "10px",
          padding: "10px",
        }}
      >
        Click on a shortcut to re-bind it.
      </div>
    </div>
  );
};
