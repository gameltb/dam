import { ChevronRight, Home } from "lucide-react";
import React from "react";
import { useShallow } from "zustand/react/shallow";

import { useFlowStore } from "@/store/flowStore";
import { useNavigationStore } from "@/store/ui/navigationStore";
import { type AppNode } from "@/types";

export const Breadcrumbs: React.FC = () => {
  const { activeScopeId, setActiveScope } = useNavigationStore(
    useShallow((s) => ({
      activeScopeId: s.activeScopeId,
      setActiveScope: s.setActiveScope,
    })),
  );

  const nodesById = useFlowStore(useShallow((s) => s.nodesById));

  const pathNodes = React.useMemo(() => {
    if (!activeScopeId) return [];

    const items = [];
    let _currId: null | string = activeScopeId;

    const visited = new Set<string>();
    let depth = 0;
    const MAX_DEPTH = 30;

    while (_currId && depth < MAX_DEPTH) {
      if (visited.has(_currId)) {
        console.error(`[Breadcrumbs] Cycle detected at node: ${_currId}. Breaking path.`);
        break;
      }
      visited.add(_currId);
      depth++;

      const node: AppNode | undefined = nodesById[_currId];
      if (node) {
        items.unshift({
          id: node.id,
          label: (node.data as { displayName?: string }).displayName ?? node.id,
          parentId: node.parentId ?? null,
        });

        const nextPId: null | string = node.parentId ?? null;
        if (nextPId === _currId) break;
        _currId = nextPId;
      } else {
        // Show placeholder if node is not loaded
        items.unshift({
          id: _currId,
          label: `Node (${_currId.slice(0, 4)}…)`,
          parentId: null,
        });
        break;
      }
    }
    return items;
  }, [activeScopeId, nodesById]);

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

      {pathNodes.map((item) => (
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
