import React, { useState } from "react";
import { useTable } from "spacetimedb/react";
import { tables } from "@/generated/spacetime";
import { ScrollArea } from "../ui/scroll-area";
import { Database, Search, Share2, MessageCircle, FileText, Wind, Sliders, Radio, Maximize, Logs, Link, CheckCircle, Box } from "lucide-react";
import { cn } from "@/lib/utils";

const AVAILABLE_TABLES = [
  { id: "operationLogs", label: "operation_logs", icon: Logs },
  { id: "clientTaskAssignments", label: "client_task_assignments", icon: Link },
  { id: "tasks", label: "tasks", icon: CheckCircle },
  { id: "nodes", label: "nodes", icon: Box },
  { id: "edges", label: "edges", icon: Share2 },
  { id: "chatMessages", label: "chat_messages", icon: MessageCircle },
  { id: "chatContents", label: "chat_contents", icon: FileText },
  { id: "chatStreams", label: "chat_streams", icon: Wind },
  { id: "widgetValues", label: "widget_values", icon: Sliders },
  { id: "nodeSignals", label: "node_signals", icon: Radio },
  { id: "viewportState", label: "viewport_state", icon: Maximize },
];

/**
 * Isolated viewer component for a single table to ensure clean useTable subscription.
 */
const TableDataView: React.FC<{ tableId: string }> = ({ tableId }) => {
  const tableHandle = (tables as any)[tableId];
  // useTable returns [rows, handle]
  const [rows] = useTable(tableHandle) as [any[], any];

  if (!rows || rows.length === 0) {
    return (
      <div className="p-12 text-center text-muted-foreground text-xs italic">
        Table "{tableId}" is empty or not yet synchronized.
      </div>
    );
  }

  return (
    <div className="p-0 min-w-full">
      <table className="w-full text-[10px] font-mono border-collapse">
        <thead>
          <tr className="bg-muted/30 sticky top-0 border-b border-border shadow-sm">
            {Object.keys(rows[0]).map((key) => (
              <th key={key} className="px-4 py-2 text-left font-bold border-r border-border last:border-r-0 whitespace-nowrap">
                {key}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className="border-b border-border/50 hover:bg-primary/5 transition-colors">
              {Object.values(row).map((val: any, j) => (
                <td key={j} className="px-4 py-1.5 whitespace-nowrap overflow-hidden text-ellipsis max-w-[250px] border-r border-border/20 last:border-r-0">
                  {formatCellValue(val)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export const SpacetimeTableBrowser: React.FC = () => {
  const [selectedTableId, setSelectedTableId] = useState("operationLogs");

  return (
    <div className="flex h-full overflow-hidden">
      {/* Table Sidebar */}
      <div className="w-52 border-r border-border bg-muted/5 flex flex-col shrink-0">
        <div className="p-3 border-b border-border flex items-center gap-2">
          <Database size={14} className="text-primary" />
          <span className="text-[10px] font-bold uppercase tracking-widest">Schemas</span>
        </div>
        <ScrollArea className="flex-1">
          <div className="p-1">
            {AVAILABLE_TABLES.map((t) => {
              const Icon = t.icon;
              return (
                <div
                  key={t.id}
                  className={cn(
                    "flex items-center gap-2 px-3 py-2 rounded-md cursor-pointer transition-colors mb-1 group",
                    selectedTableId === t.id 
                      ? "bg-primary text-primary-foreground shadow-sm" 
                      : "hover:bg-muted text-muted-foreground"
                  )}
                  onClick={() => setSelectedTableId(t.id)}
                >
                  <Icon size={12} className={cn(selectedTableId === t.id ? "text-primary-foreground" : "text-primary/60 group-hover:text-primary")} />
                  <span className="text-[10px] font-medium truncate">{t.label}</span>
                </div>
              );
            })}
          </div>
        </ScrollArea>
      </div>

      {/* Content Area */}
      <div className="flex-1 flex flex-col min-w-0 bg-background">
        <div className="p-3 border-b border-border bg-muted/10 flex justify-between items-center">
          <div className="flex items-center gap-2">
            <span className="text-xs font-bold font-mono text-primary uppercase">{selectedTableId}</span>
          </div>
          <div className="flex items-center gap-2 px-2 py-1 bg-background border rounded text-[10px] text-muted-foreground shadow-inner">
            <Search size={10} />
            <input 
              placeholder="Search..." 
              className="bg-transparent border-none outline-none w-32"
            />
          </div>
        </div>

        <ScrollArea className="flex-1 overflow-auto">
          {/* We use tableId as KEY to force complete component remount on switch,
              ensuring useTable re-subscribes correctly to the new table handle. */}
          <TableDataView key={selectedTableId} tableId={selectedTableId} />
        </ScrollArea>
      </div>
    </div>
  );
};

function formatCellValue(val: any): string {
  if (val === null || val === undefined) return "NULL";
  if (typeof val === "bigint") return val.toString();
  if (typeof val === "object") {
    try {
      return JSON.stringify(val);
    } catch {
      return "[Object]";
    }
  }
  return String(val);
}