import { Settings, X } from "lucide-react";
import { useCallback, useState } from "react";
import { useShallow } from "zustand/react/shallow";

import { type ShortcutConfig, useSettingsStore } from "@/store/ui/settingsStore";
import { useUiStore } from "@/store/uiStore";

import { AiSettings } from "./settings/AiSettings";
import { GeneralSettings } from "./settings/GeneralSettings";
import { ShortcutSettings } from "./settings/ShortcutSettings";

export function SettingsModal() {
  const { dragMode, isOpen, setDragMode, setOpen } = useUiStore(
    useShallow((s) => ({
      dragMode: s.dragMode,
      isOpen: s.isSettingsOpen,
      setDragMode: s.setDragMode,
      setOpen: s.setSettingsOpen,
    })),
  );

  const {
    activeLocalClientId,
    hotkeys: shortcuts,
    serverAddress,
    setSettings,
    showControls,
    showMinimap,
    theme,
    updateHotkeys,
  } = useSettingsStore(
    useShallow((s) => ({
      activeLocalClientId: s.activeLocalClientId,
      hotkeys: s.hotkeys,
      serverAddress: s.serverAddress,
      setSettings: s.setSettings,
      showControls: s.showControls,
      showMinimap: s.showMinimap,
      theme: s.theme,
      updateHotkeys: s.updateHotkeys,
    })),
  );

  const settings = { activeLocalClientId, serverAddress, showControls, showMinimap, theme };

  const setShortcut = useCallback(
    (key: keyof ShortcutConfig, val: string) => {
      updateHotkeys({ [key]: val });
    },
    [updateHotkeys],
  );

  const [activeTab, setActiveTab] = useState<"ai" | "general" | "shortcuts">("general");

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-[10000] flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={() => {
        setOpen(false);
      }}
    >
      <div
        className="flex flex-col w-[500px] max-h-[80vh] overflow-hidden bg-panel-bg border border-node-border rounded-xl shadow-2xl"
        onClick={(e) => {
          e.stopPropagation();
        }}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4 px-5 border-b border-node-border">
          <div className="flex items-center gap-2.5">
            <Settings className="text-primary" size={18} />
            <span className="text-base font-semibold">Settings</span>
          </div>
          <button
            className="p-1 text-muted-foreground hover:text-foreground transition-colors"
            onClick={() => {
              setOpen(false);
            }}
          >
            <X size={20} />
          </button>
        </div>

        {/* Tabs */}
        <div className="flex px-2">
          {(["general", "ai", "shortcuts"] as const).map((tab) => (
            <button
              className={`px-4 py-3 text-sm font-medium border-b-2 transition-colors capitalize ${
                activeTab === tab
                  ? "border-primary text-foreground"
                  : "border-transparent text-muted-foreground hover:text-foreground"
              }`}
              key={tab}
              onClick={() => {
                setActiveTab(tab);
              }}
            >
              {tab === "ai" ? "AI Local" : tab}
            </button>
          ))}
        </div>

        {/* Content */}
        <div className="flex-1 p-5 overflow-y-auto">
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
}
