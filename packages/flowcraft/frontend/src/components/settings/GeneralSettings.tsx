import { Keyboard, Moon, MousePointer2, Sun } from "lucide-react";
import React from "react";

import { type UISettings } from "@/store/uiStore";
import { DragMode, Theme } from "@/types";

interface GeneralSettingsProps {
  dragMode: DragMode;
  setDragMode: (mode: DragMode) => void;
  setSettings: (settings: Partial<UISettings>) => void;
  settings: UISettings;
}

export const GeneralSettings: React.FC<GeneralSettingsProps> = ({
  dragMode,
  setDragMode,
  setSettings,
  settings,
}) => {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
      {/* Server Address */}
      <div>
        <label
          style={{
            color: "var(--sub-text)",
            display: "block",
            fontSize: "13px",
            fontWeight: 500,
            marginBottom: "8px",
          }}
        >
          Server Address
        </label>
        <input
          onChange={(e) => {
            setSettings({ serverAddress: e.target.value });
          }}
          placeholder="http://localhost:3000"
          style={{
            backgroundColor: "rgba(255,255,255,0.03)",
            border: "1px solid var(--node-border)",
            borderRadius: "8px",
            color: "var(--text-color)",
            fontSize: "13px",
            outline: "none",
            padding: "10px 12px",
            width: "100%",
          }}
          type="text"
          value={settings.serverAddress}
        />
        <p
          style={{
            color: "var(--sub-text)",
            fontSize: "11px",
            marginTop: "6px",
          }}
        >
          The base URL of the gRPC/Connect backend.
        </p>
      </div>

      {/* Drag Mode */}
      <div>
        <label
          style={{
            color: "var(--sub-text)",
            display: "block",
            fontSize: "13px",
            fontWeight: 500,
            marginBottom: "12px",
          }}
        >
          Canvas Interaction
        </label>
        <div style={{ display: "flex", gap: "12px" }}>
          <button
            onClick={() => {
              setDragMode(DragMode.PAN);
            }}
            style={{
              alignItems: "center",
              backgroundColor:
                dragMode === DragMode.PAN
                  ? "rgba(100, 108, 255, 0.1)"
                  : "rgba(255,255,255,0.03)",
              border: `1px solid ${dragMode === DragMode.PAN ? "var(--primary-color)" : "var(--node-border)"}`,
              borderRadius: "8px",
              color:
                dragMode === DragMode.PAN
                  ? "var(--primary-color)"
                  : "var(--text-color)",
              cursor: "pointer",
              display: "flex",
              flex: 1,
              flexDirection: "column",
              gap: "8px",
              padding: "12px",
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
              setDragMode(DragMode.SELECT);
            }}
            style={{
              alignItems: "center",
              backgroundColor:
                dragMode === DragMode.SELECT
                  ? "rgba(100, 108, 255, 0.1)"
                  : "rgba(255,255,255,0.03)",
              border: `1px solid ${dragMode === DragMode.SELECT ? "var(--primary-color)" : "var(--node-border)"}`,
              borderRadius: "8px",
              color:
                dragMode === DragMode.SELECT
                  ? "var(--primary-color)"
                  : "var(--text-color)",
              cursor: "pointer",
              display: "flex",
              flex: 1,
              flexDirection: "column",
              gap: "8px",
              padding: "12px",
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
            color: "var(--sub-text)",
            display: "block",
            fontSize: "13px",
            fontWeight: 500,
            marginBottom: "12px",
          }}
        >
          Appearance
        </label>
        <div style={{ display: "flex", gap: "12px" }}>
          <button
            onClick={() => {
              setSettings({ theme: Theme.DARK });
            }}
            style={{
              alignItems: "center",
              backgroundColor:
                settings.theme === Theme.DARK
                  ? "rgba(100, 108, 255, 0.1)"
                  : "rgba(255,255,255,0.03)",
              border: `1px solid ${settings.theme === Theme.DARK ? "var(--primary-color)" : "var(--node-border)"}`,
              borderRadius: "8px",
              color:
                settings.theme === Theme.DARK
                  ? "var(--primary-color)"
                  : "var(--text-color)",
              cursor: "pointer",
              display: "flex",
              flex: 1,
              gap: "8px",
              justifyContent: "center",
              padding: "12px",
              transition: "all 0.2s",
            }}
          >
            <Moon size={18} />
            <span style={{ fontSize: "12px", fontWeight: 500 }}>Dark Mode</span>
          </button>
          <button
            onClick={() => {
              setSettings({ theme: Theme.LIGHT });
            }}
            style={{
              alignItems: "center",
              backgroundColor:
                settings.theme === Theme.LIGHT
                  ? "rgba(100, 108, 255, 0.1)"
                  : "rgba(255,255,255,0.03)",
              border: `1px solid ${settings.theme === Theme.LIGHT ? "var(--primary-color)" : "var(--node-border)"}`,
              borderRadius: "8px",
              color:
                settings.theme === Theme.LIGHT
                  ? "var(--primary-color)"
                  : "var(--text-color)",
              cursor: "pointer",
              display: "flex",
              flex: 1,
              gap: "8px",
              justifyContent: "center",
              padding: "12px",
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
  );
};
