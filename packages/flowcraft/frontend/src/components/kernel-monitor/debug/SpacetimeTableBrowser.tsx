import {
  Box,
  CheckCircle,
  Database,
  Link,
  Logs,
  Maximize,
  MessageCircle,
  Radio,
  Search,
  Share2,
  Sliders,
  Table as TableIcon,
  Wind,
} from "lucide-react";
import React, { useMemo, useState } from "react";

import { ScrollArea } from "@/components/ui/scroll-area";
import { tables } from "@/generated/spacetime";
import { cn } from "@/lib/utils";

import { TableDataView } from "./TableDataView";

// Icon mapping for known tables
const ICON_MAP: Record<string, React.ElementType> = {
  chat_messages: MessageCircle,
  chat_streams: Wind,
  client_task_assignments: Link,
  edges: Share2,
  node_signals: Radio,
  nodes: Box,
  operation_logs: Logs,
  tasks: CheckCircle,
  viewport_state: Maximize,
  widget_values: Sliders,
};

type TableId = keyof typeof tables;

export const SpacetimeTableBrowser: React.FC = () => {
  // Default to 'operation_logs' if it exists, otherwise the first table
  const [selectedTableId, setSelectedTableId] = useState<TableId>(
    "operationLogs" in tables ? ("operationLogs" as TableId) : (Object.keys(tables)[0] as TableId),
  );
  const [filterText, setFilterText] = useState("");

  const tableList = useMemo(() => {
    return Object.keys(tables)
      .map((key) => {
        // SpacetimeDB generated keys are usually camelCase (e.g. 'chatMessages')
        // but the actual table name might be different in the map?
        // Checking generated code: `export const tables = ...`
        // The keys in `tables` object match the accessor names.
        // Let's rely on the accessor name.
        const id = key as TableId;
        // Simple heuristic to guess snake_case name for icon lookup if needed,
        // or just use the accessor name. The previous code used snake_case labels.
        // Let's try to map accessor 'chatMessages' to 'chat_messages' for icon lookup if direct match fails.
        const snakeName = key.replace(/[A-Z]/g, (letter) => `_${letter.toLowerCase()}`);
        const Icon = ICON_MAP[key] || ICON_MAP[snakeName] || TableIcon;

        return {
          icon: Icon,
          id,
          label: snakeName, // Display snake_case as it looks more like DB table
        };
      })
      .sort((a, b) => a.label.localeCompare(b.label));
  }, []);

  return (
    <div className="flex h-full w-full min-w-0 overflow-hidden">
      {/* Table Sidebar */}
      <div className="flex w-52 shrink-0 flex-col border-r border-border bg-muted/5">
        <div className="flex items-center gap-2 border-b border-border p-3">
          <Database className="text-primary" size={14} />
          <span className="text-[10px] font-bold uppercase tracking-widest">Schemas</span>
        </div>
        <ScrollArea className="flex-1">
          <div className="p-1">
            {tableList.map((t) => {
              const Icon = t.icon;
              return (
                <div
                  className={cn(
                    "group mb-1 flex cursor-pointer items-center gap-2 rounded-md px-3 py-2 transition-colors",
                    selectedTableId === t.id
                      ? "bg-primary text-primary-foreground shadow-sm"
                      : "text-muted-foreground hover:bg-muted",
                  )}
                  key={t.id}
                  onClick={() => {
                    setSelectedTableId(t.id);
                  }}
                >
                  <Icon
                    className={cn(
                      selectedTableId === t.id ? "text-primary-foreground" : "text-primary/60 group-hover:text-primary",
                    )}
                    size={12}
                  />
                  <span className="truncate text-[10px] font-medium">{t.label}</span>
                </div>
              );
            })}
          </div>
        </ScrollArea>
      </div>

      {/* Content Area */}
      <div className="flex flex-1 flex-col overflow-hidden bg-background min-w-0 max-w-full">
        <div className="flex shrink-0 items-center justify-between border-b border-border bg-muted/10 p-3">
          <div className="flex items-center gap-2">
            <span className="font-mono text-xs font-bold uppercase text-primary">{selectedTableId}</span>
          </div>
          <div className="flex items-center gap-2 rounded border bg-background px-2 py-1 text-[10px] text-muted-foreground shadow-inner">
            <Search size={10} />
            <input
              className="w-32 border-none bg-transparent outline-none"
              onChange={(e) => {
                setFilterText(e.target.value);
              }}
              placeholder="Search…"
              value={filterText}
            />
          </div>
        </div>

        <div className="flex flex-1 flex-col overflow-hidden min-h-0">
          {/* We use tableId as KEY to force complete component remount on switch,
              ensuring useTable re-subscribes correctly to the new table handle. */}
          <TableDataView filterText={filterText} key={selectedTableId} tableId={selectedTableId} />
        </div>
      </div>
    </div>
  );
};
