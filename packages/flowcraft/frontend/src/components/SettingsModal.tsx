import { Settings, X } from "lucide-react";
import React, { useState } from "react";
import { useShallow } from "zustand/react/shallow";

import { useUiStore } from "@/store/uiStore";

import { AiSettings } from "./settings/AiSettings";
import { GeneralSettings } from "./settings/GeneralSettings";
import { ShortcutSettings } from "./settings/ShortcutSettings";

export const SettingsModal: React.FC = () => {
  const { dragMode, isOpen, setDragMode, setOpen, setSettings, setShortcut, settings, shortcuts } = useUiStore(
    useShallow((s) => ({
      dragMode: s.dragMode,
      isOpen: s.isSettingsOpen,
      setDragMode: s.setDragMode,
      setOpen: s.setSettingsOpen,
      setSettings: s.setSettings,
      setShortcut: s.setShortcut,
      settings: s.settings,
      shortcuts: s.shortcuts,
    })),
  );

  const [activeTab, setActiveTab] = useState<"ai" | "general" | "shortcuts">("general");

  if (!isOpen) return null;

  return (
    <div
      onClick={() => {
        setOpen(false);
      }}
      style={{
        alignItems: "center",
        backdropFilter: "blur(4px)",
        backgroundColor: "rgba(0, 0, 0, 0.6)",
        bottom: 0,
        display: "flex",
        justifyContent: "center",
        left: 0,
        position: "fixed",
        right: 0,
        top: 0,
        zIndex: 10000,
      }}
    >
      <div
        onClick={(e) => {
          e.stopPropagation();
        }}
        style={{
          backgroundColor: "var(--panel-bg)",
          border: "1px solid var(--node-border)",
          borderRadius: "12px",
          boxShadow: "0 20px 50px rgba(0,0,0,0.5)",
          display: "flex",
          flexDirection: "column",
          maxHeight: "80vh",
          overflow: "hidden",
          width: "500px",
        }}
      >
        {/* Header */}
        <div
          style={{
            alignItems: "center",
            borderBottom: "1px solid var(--node-border)",
            display: "flex",
            justifyContent: "space-between",
            padding: "16px 20px",
          }}
        >
          <div style={{ alignItems: "center", display: "flex", gap: "10px" }}>
            <Settings color="var(--primary-color)" size={18} />
            <span style={{ fontSize: "16px", fontWeight: 600 }}>Settings</span>
          </div>
          <button
            onClick={() => {
              setOpen(false);
            }}
            style={{
              background: "none",
              border: "none",
              color: "var(--sub-text)",
              cursor: "pointer",
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
              background: "none",
              border: "none",
              borderBottom: `2px solid ${activeTab === "general" ? "var(--primary-color)" : "transparent"}`,
              color: activeTab === "general" ? "var(--text-color)" : "var(--sub-text)",
              cursor: "pointer",
              fontSize: "13px",
              fontWeight: 500,
              padding: "12px 15px",
            }}
          >
            General
          </button>
          <button
            onClick={() => {
              setActiveTab("ai");
            }}
            style={{
              background: "none",
              border: "none",
              borderBottom: `2px solid ${activeTab === "ai" ? "var(--primary-color)" : "transparent"}`,
              color: activeTab === "ai" ? "var(--text-color)" : "var(--sub-text)",
              cursor: "pointer",
              fontSize: "13px",
              fontWeight: 500,
              padding: "12px 15px",
            }}
          >
            AI Local
          </button>
          <button
            onClick={() => {
              setActiveTab("shortcuts");
            }}
            style={{
              background: "none",
              border: "none",
              borderBottom: `2px solid ${activeTab === "shortcuts" ? "var(--primary-color)" : "transparent"}`,
              color: activeTab === "shortcuts" ? "var(--text-color)" : "var(--sub-text)",
              cursor: "pointer",
              fontSize: "13px",
              fontWeight: 500,
              padding: "12px 15px",
            }}
          >
            Shortcuts
          </button>
        </div>

        {/* Content */}
        <div style={{ flex: 1, overflowY: "auto", padding: "20px" }}>
          {activeTab === "general" && (
            <GeneralSettings
              dragMode={dragMode}
              setDragMode={setDragMode}
              setSettings={setSettings}
              settings={settings}
            />
          )}
          {activeTab === "ai" && <AiSettings />}
          {activeTab === "shortcuts" && <ShortcutSettings setShortcut={setShortcut} shortcuts={shortcuts} />}
        </div>
      </div>
    </div>
  );
};
