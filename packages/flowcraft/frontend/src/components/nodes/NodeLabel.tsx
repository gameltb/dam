import React, { memo, useState } from "react";

import { useNodeProperty } from "@/hooks/core/useNodeProperty";

interface NodeLabelProps {
  selected?: boolean;
}

export const NodeLabel: React.FC<NodeLabelProps> = memo(({ selected }) => {
  // Bind directly to 'displayName' using the context-aware property hook
  const [displayName, setDisplayName] = useNodeProperty("displayName");

  const [isEditing, setIsEditing] = useState(false);
  const [localValue, setLocalValue] = useState(displayName ?? "");

  const handleBlur = () => {
    setIsEditing(false);
    if (localValue !== displayName) {
      setDisplayName(localValue);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      (e.target as HTMLInputElement).blur();
    }
  };

  if (!isEditing) {
    return (
      <div
        className={`px-3 py-2 text-xs font-bold uppercase tracking-tight truncate border-b border-node-border bg-muted/30 cursor-text hover:bg-muted/50 transition-colors ${selected ? "text-primary" : "text-muted-foreground"}`}
        onDoubleClick={() => {
          setLocalValue(displayName);
          setIsEditing(true);
        }}
      >
        {displayName || "Untitled Node"}
      </div>
    );
  }

  return (
    <div className="px-3 py-2 border-b border-node-border bg-background">
      <input
        autoFocus
        className="w-full bg-transparent text-xs font-bold uppercase outline-none text-primary"
        onBlur={handleBlur}
        onChange={(e) => {
          setLocalValue(e.target.value);
        }}
        onKeyDown={handleKeyDown}
        value={localValue}
      />
    </div>
  );
});
