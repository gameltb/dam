import React, { memo, useState } from "react";
import { useShallow } from "zustand/react/shallow";

import { useFlowStore } from "@/store/flowStore";

interface NodeLabelProps {
  id: string;
  label: string;
  onChange: (id: string, label: string) => void;
  selected?: boolean;
}

export const NodeLabel: React.FC<NodeLabelProps> = memo(({ id, label, selected }) => {
  const { allNodes, nodeDraft } = useFlowStore(
    useShallow((s) => ({
      allNodes: s.allNodes,
      nodeDraft: s.nodeDraft,
    })),
  );
  const [isEditing, setIsEditing] = useState(false);
  const [value, setValue] = useState(label);

  const handleBlur = () => {
    setIsEditing(false);
    if (value !== label) {
      const node = allNodes.find((n) => n.id === id);
      if (node) {
        const res = nodeDraft(node);
        if (res.ok) {
          res.value.data.displayName = value;
        }
      }
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
          setIsEditing(true);
        }}
      >
        {label || "Untitled Node"}
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
          setValue(e.target.value);
        }}
        onKeyDown={handleKeyDown}
        value={value}
      />
    </div>
  );
});
