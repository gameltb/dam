import React, { useEffect, useState } from "react";
import { type ShortcutConfig } from "@/store/ui/settingsStore";

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
      className={`px-2 py-1 min-w-[80px] font-mono text-[11px] rounded border transition-colors ${
        isRecording 
          ? "bg-primary/20 border-primary text-primary" 
          : "bg-white/5 border-white/10 text-foreground"
      }`}
      onClick={() => {
        setIsRecording(true);
      }}
    >
      {isRecording ? "Press keys…" : value}
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
    <div className="flex flex-col gap-3">
      {shortcutList.map((s, i) => (
        <div
          key={i}
          className="flex items-center justify-between py-2 border-b border-white/5"
        >
          <span className="text-sm">{s.label}</span>
          <ShortcutRecordButton
            label={s.label}
            onSave={(val) => {
              setShortcut(s.key, val);
            }}
            value={shortcuts[s.key]}
          />
        </div>
      ))}

      <div className="mt-2 p-2.5 bg-primary/5 rounded text-[11px] text-muted-foreground italic">
        Click on a shortcut to re-bind it.
      </div>
    </div>
  );
};