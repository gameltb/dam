import {
  ArrowDown,
  ArrowUp,
  Box,
  CheckCircle,
  ChevronLeft,
  ChevronRight,
  Database,
  Link,
  Logs,
  Maximize,
  MessageCircle,
  Radio,
  Search,
  Share2,
  Sliders,
  Wind,
} from "lucide-react";
import React, { useMemo, useState } from "react";
import { useTable } from "spacetimedb/react";

import { tables } from "@/generated/spacetime";
import { cn } from "@/lib/utils";

import { ScrollArea } from "../ui/scroll-area";

const AVAILABLE_TABLES = [
  { icon: Logs, id: "operationLogs", label: "operation_logs" },
  { icon: Link, id: "clientTaskAssignments", label: "client_task_assignments" },
  { icon: CheckCircle, id: "tasks", label: "tasks" },
  { icon: Box, id: "nodes", label: "nodes" },
  { icon: Share2, id: "edges", label: "edges" },
  { icon: MessageCircle, id: "chatMessages", label: "chat_messages" },
  { icon: Wind, id: "chatStreams", label: "chat_streams" },
  { icon: Sliders, id: "widgetValues", label: "widget_values" },
  { icon: Radio, id: "nodeSignals", label: "node_signals" },
  { icon: Maximize, id: "viewportState", label: "viewport_state" },
] as const;

interface SortConfig {
  direction: "asc" | "desc";
  key: string;
}

type TableId = (typeof AVAILABLE_TABLES)[number]["id"];

/**
 * Isolated viewer component for a single table to ensure clean useTable subscription.
 */
const TableDataView: React.FC<{ filterText: string; tableId: TableId }> = ({ filterText, tableId }) => {
  const tableHandle = (tables as any)[tableId];
  const [rows] = useTable(tableHandle);

  const [sortConfig, setSortConfig] = useState<null | SortConfig>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const pageSize = 25;

  // Reset to page 1 when filter changes
  React.useEffect(() => {
    setCurrentPage(1);
  }, [filterText]);

  const handleSort = (key: string) => {
    setSortConfig((prev) => {
      if (prev?.key === key) {
        if (prev.direction === "asc") return { direction: "desc", key };
        return null;
      }
      return { direction: "asc", key };
    });
  };

  const processedRows = useMemo(() => {
    if (rows.length === 0) return [];

    let result = [...rows] as Record<string, unknown>[];

    // 1. Filter
    if (filterText) {
      const lowerFilter = filterText.toLowerCase();
      result = result.filter((row) =>
        Object.values(row).some((val) => formatCellValue(val).toLowerCase().includes(lowerFilter)),
      );
    }

    // 2. Sort
    if (sortConfig) {
      result.sort((a, b) => {
        const valA = a[sortConfig.key];
        const valB = b[sortConfig.key];

        if (valA === valB) return 0;
        if (valA === null || valA === undefined) return 1;
        if (valB === null || valB === undefined) return -1;

        if (typeof valA === "number" && typeof valB === "number") {
          return sortConfig.direction === "asc" ? valA - valB : valB - valA;
        }

        const strA = formatCellValue(valA);
        const strB = formatCellValue(valB);
        const comparison = strA.localeCompare(strB);
        return sortConfig.direction === "asc" ? comparison : -comparison;
      });
    }

    return result;
  }, [rows, filterText, sortConfig]);

  const paginatedRows = useMemo(() => {
    const start = (currentPage - 1) * pageSize;
    return processedRows.slice(start, start + pageSize);
  }, [processedRows, currentPage]);

  const totalPages = Math.ceil(processedRows.length / pageSize);

  if (rows.length === 0) {
    return (
      <div className="p-12 text-center text-muted-foreground text-xs italic">
        Table "{tableId}" is empty or not yet synchronized.
      </div>
    );
  }

  const firstRow = rows[0] as Record<string, unknown>;
  const keys = Object.keys(firstRow);

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <ScrollArea className="flex-1 w-full" type="always">
        <div className="min-w-max">
          <table className="text-[10px] font-mono border-collapse relative min-w-full">
            <thead className="sticky top-0 z-30 bg-background shadow-sm">
              <tr className="bg-muted shadow-sm">
                {keys.map((key) => (
                  <th
                    className="px-4 py-2 text-left font-bold border-r border-border last:border-r-0 whitespace-nowrap cursor-pointer hover:bg-muted-foreground/10 transition-colors select-none"
                    key={key}
                    onClick={() => {
                      handleSort(key);
                    }}
                  >
                    <div className="flex items-center gap-1">
                      {key}
                      {sortConfig?.key === key &&
                        (sortConfig.direction === "asc" ? <ArrowUp size={10} /> : <ArrowDown size={10} />)}
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {paginatedRows.map((row, i) => (
                <tr className="border-b border-border/50 hover:bg-primary/5 transition-colors" key={i}>
                  {keys.map((key, j) => (
                    <td
                      className="px-4 py-1.5 whitespace-nowrap overflow-hidden text-ellipsis max-w-[400px] border-r border-border/20 last:border-r-0"
                      key={j}
                    >
                      {formatCellValue(row[key])}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </ScrollArea>

      {/* Pagination Controls */}
      <div className="p-2 border-t border-border bg-muted/5 flex items-center justify-between text-[10px]">
        <div className="text-muted-foreground">
          Showing {Math.min(processedRows.length, (currentPage - 1) * pageSize + 1)} -{" "}
          {Math.min(processedRows.length, currentPage * pageSize)} of {processedRows.length} rows
        </div>
        <div className="flex items-center gap-2">
          <button
            className="p-1 hover:bg-muted rounded disabled:opacity-30"
            disabled={currentPage === 1}
            onClick={() => {
              setCurrentPage((p) => p - 1);
            }}
          >
            <ChevronLeft size={14} />
          </button>
          <span className="font-bold">
            Page {currentPage} of {totalPages || 1}
          </span>
          <button
            className="p-1 hover:bg-muted rounded disabled:opacity-30"
            disabled={currentPage >= totalPages}
            onClick={() => {
              setCurrentPage((p) => p + 1);
            }}
          >
            <ChevronRight size={14} />
          </button>
        </div>
      </div>
    </div>
  );
};

export const SpacetimeTableBrowser: React.FC = () => {
  const [selectedTableId, setSelectedTableId] = useState<TableId>("operationLogs");
  const [filterText, setFilterText] = useState("");

  return (
    <div className="flex h-full overflow-hidden">
      {/* Table Sidebar */}
      <div className="w-52 border-r border-border bg-muted/5 flex flex-col shrink-0">
        <div className="p-3 border-b border-border flex items-center gap-2">
          <Database className="text-primary" size={14} />
          <span className="text-[10px] font-bold uppercase tracking-widest">Schemas</span>
        </div>
        <ScrollArea className="flex-1">
          <div className="p-1">
            {AVAILABLE_TABLES.map((t) => {
              const Icon = t.icon;
              return (
                <div
                  className={cn(
                    "flex items-center gap-2 px-3 py-2 rounded-md cursor-pointer transition-colors mb-1 group",
                    selectedTableId === t.id
                      ? "bg-primary text-primary-foreground shadow-sm"
                      : "hover:bg-muted text-muted-foreground",
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
              className="bg-transparent border-none outline-none w-32"
              onChange={(e) => {
                setFilterText(e.target.value);
              }}
              placeholder="Search..."
              value={filterText}
            />
          </div>
        </div>

        <div className="flex-1 min-h-0 overflow-hidden">
          {/* We use tableId as KEY to force complete component remount on switch,
              ensuring useTable re-subscribes correctly to the new table handle. */}
          <TableDataView filterText={filterText} key={selectedTableId} tableId={selectedTableId} />
        </div>
      </div>
    </div>
  );
};

function formatCellValue(val: unknown): string {
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
