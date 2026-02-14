import { Keyboard, Moon, MousePointer2, Sun } from "lucide-react";
import React from "react";

import { type SettingsState } from "@/store/ui/settingsStore";
import { DragMode, Theme } from "@/types";

interface GeneralSettingsProps {
  dragMode: DragMode;
  setDragMode: (mode: DragMode) => void;
  setSettings: (settings: Partial<SettingsState>) => void;
  settings: Partial<SettingsState>;
}

export const GeneralSettings: React.FC<GeneralSettingsProps> = ({ dragMode, setDragMode, setSettings, settings }) => {
  return (
    <div className="flex flex-col gap-6">
      {/* Server Address */}
      <div>
        <label className="block mb-2 text-sm font-medium text-muted-foreground">Server Address</label>
        <input
          className="w-full px-3 py-2 text-sm border rounded-lg bg-white/5 border-node-border text-foreground outline-none focus:border-primary"
          onChange={(e) => {
            setSettings({ serverAddress: e.target.value });
          }}
          placeholder="http://localhost:3000"
          type="text"
          value={settings.serverAddress}
        />
        <p className="mt-1.5 text-[11px] text-muted-foreground">The base URL of the gRPC/Connect backend.</p>
      </div>

      {/* Drag Mode */}
      <div>
        <label className="block mb-3 text-sm font-medium text-muted-foreground">Canvas Interaction</label>
        <div className="flex gap-3">
          {[
            { icon: MousePointer2, id: DragMode.PAN, label: "Panning" },
            { icon: Keyboard, id: DragMode.SELECT, label: "Selection" },
          ].map(({ icon: Icon, id, label }) => (
            <button
              className={`flex-1 flex flex-col items-center gap-2 p-3 rounded-lg border transition-all ${
                dragMode === id
                  ? "bg-primary/10 border-primary text-primary"
                  : "bg-white/5 border-node-border text-foreground"
              }`}
              key={id}
              onClick={() => {
                setDragMode(id);
              }}
            >
              <Icon size={20} />
              <span className="text-xs font-medium">{label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Appearance */}
      <div>
        <label className="block mb-3 text-sm font-medium text-muted-foreground">Appearance</label>
        <div className="flex gap-3">
          {[
            { icon: Moon, id: Theme.DARK, label: "Dark Mode" },
            { icon: Sun, id: Theme.LIGHT, label: "Light Mode" },
          ].map(({ icon: Icon, id, label }) => (
            <button
              className={`flex-1 flex items-center justify-center gap-2 p-3 rounded-lg border transition-all ${
                settings.theme === id
                  ? "bg-primary/10 border-primary text-primary"
                  : "bg-white/5 border-node-border text-foreground"
              }`}
              key={id}
              onClick={() => {
                setSettings({ theme: id });
              }}
            >
              <Icon size={18} />
              <span className="text-xs font-medium">{label}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};
