import { ChevronRight, Home } from "lucide-react";
import React from "react";
import { useShallow } from "zustand/react/shallow";

import { useFlowStore } from "@/store/flowStore";
import { useUiStore } from "@/store/uiStore";

export const Breadcrumbs: React.FC = () => {
  const { activeScopeId, setActiveScope } = useUiStore(
    useShallow((s) => ({
      activeScopeId: s.activeScopeId,
      setActiveScope: s.setActiveScope,
    })),
  );
  const allNodes = useFlowStore((s) => s.allNodes);

  const path = React.useMemo(() => {
    const items = [];
    let currentId = activeScopeId;
    while (currentId) {
      const node = allNodes.find((n) => n.id === currentId);
      if (node) {
        items.unshift({ id: node.id, label: node.data.displayName || node.id });
        currentId = node.parentId || null;
      } else {
        break;
      }
    }
    return items;
  }, [activeScopeId, allNodes]);

  return (
    <div className="absolute top-4 left-4 z-[1000] flex items-center gap-2 bg-background/80 backdrop-blur border border-border px-3 py-1.5 rounded-full shadow-lg text-xs font-medium">
      <button
        className={`flex items-center gap-1 hover:text-primary transition-colors ${!activeScopeId ? "text-primary" : "text-muted-foreground"}`}
        onClick={() => {
          setActiveScope(null);
        }}
      >
        <Home size={14} />
        <span>Root</span>
      </button>

      {path.map((item) => (
        <React.Fragment key={item.id}>
          <ChevronRight className="text-muted-foreground" size={12} />
          <button
            className={`hover:text-primary transition-colors ${activeScopeId === item.id ? "text-primary" : "text-muted-foreground"}`}
            onClick={() => {
              setActiveScope(item.id);
            }}
          >
            {item.label}
          </button>
        </React.Fragment>
      ))}
    </div>
  );
};
