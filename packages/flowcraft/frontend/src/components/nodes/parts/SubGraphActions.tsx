import { Maximize2 } from "lucide-react";
import { memo } from "react";

import { useNavigationStore } from "@/store/ui/navigationStore";

interface SubGraphActionsProps {
  id: string;
  isHovered: boolean;
}

export const SubGraphActions = memo(({ id, isHovered }: SubGraphActionsProps) => {
  const setActiveScope = useNavigationStore((s) => s.setActiveScope);

  if (!isHovered) return null;

  return (
    <div className="absolute top-2 right-2 z-50 animate-in fade-in zoom-in duration-200">
      <button
        className="flex items-center gap-1.5 px-2.5 py-1.5 bg-primary/90 hover:bg-primary text-primary-foreground rounded-md text-[10px] font-bold uppercase tracking-wider shadow-lg backdrop-blur-sm transition-all hover:scale-105 active:scale-95"
        onClick={(e) => {
          e.stopPropagation();
          setActiveScope(id);
        }}
      >
        <Maximize2 size={12} />
        Enter Subgraph
      </button>
    </div>
  );
});

SubGraphActions.displayName = "SubGraphActions";
