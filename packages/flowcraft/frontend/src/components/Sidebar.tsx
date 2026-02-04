import { memo } from "react";
import { useShallow } from "zustand/react/shallow";
import { useUiStore } from "@/store/uiStore";

export const Sidebar = memo(() => {
  const { isSidebarOpen, setSidebarOpen, sidebarWidth } = useUiStore(
    useShallow((s) => ({
      isSidebarOpen: s.isSidebarOpen,
      setSidebarOpen: s.setSidebarOpen,
      sidebarWidth: s.sidebarWidth,
    })),
  );

  return (
    <aside
      className={`relative h-full border-r border-border bg-background transition-all duration-300 ease-in-out z-40 ${
        isSidebarOpen ? "opacity-100" : "w-0 opacity-0 overflow-hidden"
      }`}
      style={{ width: isSidebarOpen ? sidebarWidth : 0 }}
    >
      <div className="flex flex-col h-full w-[400px]">
        {/* Sidebar Content */}
        <div className="p-4 border-b border-border bg-muted/20">
          <h2 className="text-sm font-bold uppercase tracking-widest opacity-50">Explorer</h2>
        </div>
        <div className="flex-1 overflow-y-auto">
          {/* Node items or tree would go here */}
        </div>
      </div>

      {/* Resize Handle (Placeholder for actual resizer) */}
      <div
        className="absolute right-0 top-0 bottom-0 w-1 cursor-col-resize hover:bg-primary/50 transition-colors"
        onDoubleClick={() => setSidebarOpen(false)}
      />
    </aside>
  );
});

Sidebar.displayName = "Sidebar";