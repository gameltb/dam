import { ArrowDown, ArrowUp, ChevronLeft, ChevronRight } from "lucide-react";
import React, { useMemo, useState } from "react";
import { useTable } from "spacetimedb/react";

import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area";
import { tables } from "@/generated/spacetime";
import { convertStdbToPbJson } from "@/utils/pb-client";

import { formatCellValue, JsonCell } from "./JsonCell";

export type TableId = keyof typeof tables;

interface SortConfig {
  direction: "asc" | "desc";
  key: string;
}

interface TableDataViewProps {
  filterText: string;
  tableId: TableId;
}

/**
 * Isolated viewer component for a single table to ensure clean useTable subscription.
 */
export const TableDataView: React.FC<TableDataViewProps> = ({ filterText, tableId }) => {
  // Safe access to table handle using strict key
  const tableHandle = tables[tableId];
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

    // Convert rows to PB JSON if mapping exists, otherwise keep as is.
    // This allows viewing the "Application View" of the data.
    let result = rows.map((r) => convertStdbToPbJson(tableId, r) as Record<string, unknown>);

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

        // Handle BigInt sorting
        if (typeof valA === "bigint" && typeof valB === "bigint") {
          // Compare BigInts directly
          if (valA < valB) return sortConfig.direction === "asc" ? -1 : 1;
          if (valA > valB) return sortConfig.direction === "asc" ? 1 : -1;
          return 0;
        }

        const strA = formatCellValue(valA);
        const strB = formatCellValue(valB);
        const comparison = strA.localeCompare(strB);
        return sortConfig.direction === "asc" ? comparison : -comparison;
      });
    }

    return result;
  }, [rows, filterText, sortConfig, tableId]);

  const paginatedRows = useMemo(() => {
    const start = (currentPage - 1) * pageSize;
    return processedRows.slice(start, start + pageSize);
  }, [processedRows, currentPage, pageSize]);

  const totalPages = Math.ceil(processedRows.length / pageSize);

  const columns = useMemo(() => {
    if (processedRows.length === 0) return [];
    // Use the first processed row to determine columns.
    const firstRow = processedRows[0];
    if (!firstRow) return [];
    return Object.keys(firstRow);
  }, [processedRows]);

  if (rows.length === 0) {
    return (
      <div className="p-12 text-center text-xs italic text-muted-foreground">
        Table "{tableId}" is empty or not yet synchronized.
      </div>
    );
  }

  return (
    <div className="flex h-full min-w-0 flex-col overflow-hidden">
      <ScrollArea className="flex-1 min-w-0 p-1" type="always">
        <div className="min-w-max">
          <table className="relative border-collapse font-mono text-[10px]">
            <thead className="sticky top-0 z-30 bg-background shadow-sm">
              <tr className="bg-muted shadow-sm">
                {columns.map((key) => (
                  <th
                    className="min-w-[150px] cursor-pointer select-none whitespace-nowrap border-r border-border px-4 py-2 text-left font-bold transition-colors last:border-r-0 hover:bg-muted-foreground/10"
                    key={key}
                    onClick={() => {
                      handleSort(key);
                    }}
                  >
                    <div className="flex items-center justify-between gap-1">
                      <span className="truncate">{key}</span>
                      {sortConfig?.key === key &&
                        (sortConfig.direction === "asc" ? <ArrowUp size={10} /> : <ArrowDown size={10} />)}
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-border/30">
              {paginatedRows.map((row, i) => (
                <tr className="group transition-colors hover:bg-primary/5" key={i}>
                  {columns.map((key, j) => (
                    <td
                      className="min-w-0 max-w-[400px] border-r border-border/20 px-4 py-1.5 align-top last:border-r-0"
                      key={j}
                    >
                      <JsonCell value={row[key]} />
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <ScrollBar className="z-50" orientation="horizontal" />
      </ScrollArea>

      {/* Pagination Controls */}
      <div className="flex shrink-0 items-center justify-between border-t border-border bg-muted/5 p-2 text-[10px]">
        <div className="text-muted-foreground">
          Showing {Math.min(processedRows.length, (currentPage - 1) * pageSize + 1)} -{" "}
          {Math.min(processedRows.length, currentPage * pageSize)} of {processedRows.length} rows
        </div>
        <div className="flex items-center gap-2">
          <button
            className="rounded p-1 transition-colors hover:bg-muted disabled:opacity-30"
            disabled={currentPage === 1}
            onClick={() => {
              setCurrentPage((p) => p - 1);
            }}
          >
            <ChevronLeft size={14} />
          </button>
          <span className="rounded border border-border/50 bg-muted/50 px-2 py-0.5 font-bold">
            Page {currentPage} of {totalPages || 1}
          </span>
          <button
            className="rounded p-1 transition-colors hover:bg-muted disabled:opacity-30"
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
