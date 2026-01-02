import React, { useState, useEffect } from "react";
import { useUiStore, type ShortcutConfig } from "../store/uiStore";
import { X, Settings, Keyboard, MousePointer2, Moon, Sun } from "lucide-react";
import { useShallow } from "zustand/react/shallow";

const ShortcutRecordButton: React.FC<{
  label: string;
  value: string;
  onSave: (val: string) => void;
}> = ({ value, onSave }) => {
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
      if (
        key !== "control" &&
        key !== "shift" &&
        key !== "alt" &&
        key !== "meta"
      ) {
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
        backgroundColor: isRecording
          ? "rgba(100, 108, 255, 0.2)"
          : "rgba(255,255,255,0.08)",
        padding: "4px 10px",
        borderRadius: "6px",
        fontSize: "11px",
        color: isRecording ? "var(--primary-color)" : "#fff",
        fontFamily: "monospace",
        border: `1px solid ${isRecording ? "var(--primary-color)" : "rgba(255,255,255,0.1)"}`,
        cursor: "pointer",
        minWidth: "80px",
        textAlign: "center",
      }}
    >
      {isRecording ? "Press keys..." : value}
    </button>
  );
};

export const SettingsModal: React.FC = () => {
  const {
    isOpen,
    setOpen,
    dragMode,
    setDragMode,
    shortcuts,
    setShortcut,
    settings,
    setSettings,
  } = useUiStore(
    useShallow((s) => ({
      isOpen: s.isSettingsOpen,
      setOpen: s.setSettingsOpen,
      dragMode: s.dragMode,
      setDragMode: s.setDragMode,
      shortcuts: s.shortcuts,
      setShortcut: s.setShortcut,
      settings: s.settings,
      setSettings: s.setSettings,
    })),
  );

  const [activeTab, setActiveTab] = useState<"general" | "shortcuts">(
    "general",
  );

  if (!isOpen) return null;

  const shortcutList: { label: string; key: keyof ShortcutConfig }[] = [
    { label: "Undo", key: "undo" },
    { label: "Redo", key: "redo" },
    { label: "Copy", key: "copy" },
    { label: "Paste", key: "paste" },
    { label: "Duplicate", key: "duplicate" },
    { label: "Delete", key: "delete" },
    { label: "Auto Layout", key: "autoLayout" },
  ];

  return (
    <div
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: "rgba(0, 0, 0, 0.6)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 10000,
        backdropFilter: "blur(4px)",
      }}
      onClick={() => {
        setOpen(false);
      }}
    >
      <div
        style={{
          width: "500px",
          backgroundColor: "var(--panel-bg)",
          borderRadius: "12px",
          border: "1px solid var(--node-border)",
          boxShadow: "0 20px 50px rgba(0,0,0,0.5)",
          display: "flex",
          flexDirection: "column",
          maxHeight: "80vh",
          overflow: "hidden",
        }}
        onClick={(e) => {
          e.stopPropagation();
        }}
      >
        {/* Header */}
        <div
          style={{
            padding: "16px 20px",
            borderBottom: "1px solid var(--node-border)",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            <Settings size={18} color="var(--primary-color)" />
            <span style={{ fontWeight: 600, fontSize: "16px" }}>Settings</span>
          </div>
          <button
            onClick={() => {
              setOpen(false);
            }}
            style={{
              background: "none",
              border: "none",
              cursor: "pointer",
              color: "var(--sub-text)",
              padding: "4px",
            }}
          >
            <X size={20} />
          </button>
        </div>

        {/* Tabs */}
        <div style={{ display: "flex", padding: "0 10px" }}>
          <button
            onClick={() => {
              setActiveTab("general");
            }}
            style={{
              padding: "12px 15px",
              background: "none",
              border: "none",
              borderBottom: `2px solid ${activeTab === "general" ? "var(--primary-color)" : "transparent"}`,
              color:
                activeTab === "general"
                  ? "var(--text-color)"
                  : "var(--sub-text)",
              cursor: "pointer",
              fontSize: "13px",
              fontWeight: 500,
            }}
          >
            General
          </button>
          <button
            onClick={() => {
              setActiveTab("shortcuts");
            }}
            style={{
              padding: "12px 15px",
              background: "none",
              border: "none",
              borderBottom: `2px solid ${activeTab === "shortcuts" ? "var(--primary-color)" : "transparent"}`,
              color:
                activeTab === "shortcuts"
                  ? "var(--text-color)"
                  : "var(--sub-text)",
              cursor: "pointer",
              fontSize: "13px",
              fontWeight: 500,
            }}
          >
            Shortcuts
          </button>
        </div>

        {/* Content */}
        <div style={{ padding: "20px", overflowY: "auto", flex: 1 }}>
          {activeTab === "general" && (
            <div
              style={{ display: "flex", flexDirection: "column", gap: "24px" }}
            >
              {/* Drag Mode */}
              <div>
                <label
                  style={{
                    display: "block",
                    marginBottom: "12px",
                    fontSize: "13px",
                    color: "var(--sub-text)",
                    fontWeight: 500,
                  }}
                >
                  Canvas Interaction
                </label>
                <div style={{ display: "flex", gap: "12px" }}>
                  <button
                    onClick={() => {
                      setDragMode("pan");
                    }}
                    style={{
                      flex: 1,
                      padding: "12px",
                      borderRadius: "8px",
                      border: `1px solid ${dragMode === "pan" ? "var(--primary-color)" : "var(--node-border)"}`,
                      backgroundColor:
                        dragMode === "pan"
                          ? "rgba(100, 108, 255, 0.1)"
                          : "rgba(255,255,255,0.03)",
                      color:
                        dragMode === "pan"
                          ? "var(--primary-color)"
                          : "var(--text-color)",
                      cursor: "pointer",
                      display: "flex",
                      flexDirection: "column",
                      alignItems: "center",
                      gap: "8px",
                      transition: "all 0.2s",
                    }}
                  >
                    <MousePointer2 size={20} />
                    <span style={{ fontSize: "12px", fontWeight: 500 }}>
                      Panning (Left Click)
                    </span>
                  </button>
                  <button
                    onClick={() => {
                      setDragMode("select");
                    }}
                    style={{
                      flex: 1,
                      padding: "12px",
                      borderRadius: "8px",
                      border: `1px solid ${dragMode === "select" ? "var(--primary-color)" : "var(--node-border)"}`,
                      backgroundColor:
                        dragMode === "select"
                          ? "rgba(100, 108, 255, 0.1)"
                          : "rgba(255,255,255,0.03)",
                      color:
                        dragMode === "select"
                          ? "var(--primary-color)"
                          : "var(--text-color)",
                      cursor: "pointer",
                      display: "flex",
                      flexDirection: "column",
                      alignItems: "center",
                      gap: "8px",
                      transition: "all 0.2s",
                    }}
                  >
                    <Keyboard size={20} />
                    <span style={{ fontSize: "12px", fontWeight: 500 }}>
                      Selection (Left Click)
                    </span>
                  </button>
                </div>
              </div>

              {/* Appearance */}
              <div>
                <label
                  style={{
                    display: "block",
                    marginBottom: "12px",
                    fontSize: "13px",
                    color: "var(--sub-text)",
                    fontWeight: 500,
                  }}
                >
                  Appearance
                </label>
                <div style={{ display: "flex", gap: "12px" }}>
                  <button
                    onClick={() => {
                      setSettings({ theme: "dark" });
                    }}
                    style={{
                      flex: 1,
                      padding: "12px",
                      borderRadius: "8px",
                      border: `1px solid ${settings.theme === "dark" ? "var(--primary-color)" : "var(--node-border)"}`,
                      backgroundColor:
                        settings.theme === "dark"
                          ? "rgba(100, 108, 255, 0.1)"
                          : "rgba(255,255,255,0.03)",
                      color:
                        settings.theme === "dark"
                          ? "var(--primary-color)"
                          : "var(--text-color)",
                      cursor: "pointer",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      gap: "8px",
                      transition: "all 0.2s",
                    }}
                  >
                    <Moon size={18} />
                    <span style={{ fontSize: "12px", fontWeight: 500 }}>
                      Dark Mode
                    </span>
                  </button>
                  <button
                    onClick={() => {
                      setSettings({ theme: "light" });
                    }}
                    style={{
                      flex: 1,
                      padding: "12px",
                      borderRadius: "8px",
                      border: `1px solid ${settings.theme === "light" ? "var(--primary-color)" : "var(--node-border)"}`,
                      backgroundColor:
                        settings.theme === "light"
                          ? "rgba(100, 108, 255, 0.1)"
                          : "rgba(255,255,255,0.03)",
                      color:
                        settings.theme === "light"
                          ? "var(--primary-color)"
                          : "var(--text-color)",
                      cursor: "pointer",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      gap: "8px",
                      transition: "all 0.2s",
                    }}
                  >
                    <Sun size={18} />
                    <span style={{ fontSize: "12px", fontWeight: 500 }}>
                      Light Mode
                    </span>
                  </button>
                </div>
              </div>
            </div>
          )}

          {activeTab === "shortcuts" && (
            <div
              style={{ display: "flex", flexDirection: "column", gap: "12px" }}
            >
              {shortcutList.map((s, i) => (
                <div
                  key={i}
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    padding: "8px 0",
                    borderBottom: "1px solid rgba(255,255,255,0.05)",
                  }}
                >
                  <span style={{ fontSize: "13px" }}>{s.label}</span>
                  <ShortcutRecordButton
                    label={s.label}
                    value={shortcuts[s.key]}
                    onSave={(val) => {
                      setShortcut(s.key, val);
                    }}
                  />
                </div>
              ))}

              <div
                style={{
                  marginTop: "10px",
                  padding: "10px",
                  borderRadius: "6px",
                  backgroundColor: "rgba(100, 108, 255, 0.05)",
                  fontSize: "11px",
                  color: "var(--sub-text)",
                  fontStyle: "italic",
                }}
              >
                Click on a shortcut to re-bind it.
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};