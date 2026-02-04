import { Code2, ListTree } from "lucide-react";
import React, { useCallback, useMemo, useState } from "react";
import { toast } from "react-hot-toast";

import { cn } from "@/lib/utils";

import { JsonTreeView } from "./JsonTreeView";

interface JsonCellProps {
  value: unknown;
}

const formatCellValueInternal = (val: unknown): string => {
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
};

export { formatCellValueInternal as formatCellValue };

/**
 * Enhanced cell component that can toggle between JSON Tree and Text view.
 */
export const JsonCell: React.FC<JsonCellProps> = ({ value }) => {
  const [viewMode, setViewMode] = useState<"text" | "tree">("text");

  // BigInts are returned as objects sometimes in STDB JS
  const isObject = value !== null && typeof value === "object" && !(typeof value === "bigint");

  const onContextMenu = useCallback(
    (e: React.MouseEvent) => {
      if (!isObject) return;
      e.preventDefault();
      setViewMode((prev) => (prev === "text" ? "tree" : "text"));
      toast.success(`Switched to ${viewMode === "text" ? "Tree" : "Text"} view`, {
        duration: 1000,
        icon: viewMode === "text" ? <ListTree size={14} /> : <Code2 size={14} />,
        id: "view-toggle",
      });
    },
    [isObject, viewMode],
  );

  const copyValue = useCallback(() => {
    void navigator.clipboard.writeText(formatCellValueInternal(value));
    toast.success("Copied to clipboard", { id: "cell-copy" });
  }, [value]);

  const jsonString = useMemo(() => {
    if (!isObject) return "";
    try {
      return JSON.stringify(value);
    } catch {
      return "[Complex Object]";
    }
  }, [value, isObject]);

  if (!isObject) {
    return (
      <div
        className="truncate group relative flex items-center min-w-0"
        onDoubleClick={copyValue}
        title="Double click to copy"
      >
        <span
          className={cn(
            "font-mono truncate",
            value === null || value === undefined ? "text-muted-foreground/40 italic" : "",
          )}
        >
          {formatCellValueInternal(value)}
        </span>
      </div>
    );
  }

  return (
    <div
      className={cn("min-w-0 transition-all overflow-hidden", viewMode === "tree" ? "py-1" : "")}
      onContextMenu={onContextMenu}
      title="Right-click to toggle Tree/Text view"
    >
      {viewMode === "tree" ? (
        <div className="bg-muted/30 p-2 rounded border border-border/50 max-h-[300px] overflow-auto min-w-[200px]">
          <JsonTreeView data={value as any} />
        </div>
      ) : (
        <div className="flex items-center gap-2 group min-w-0">
          <Code2 className="text-muted-foreground/50 shrink-0" size={12} />
          <span className="truncate font-mono text-muted-foreground/80 italic">
            {jsonString.slice(0, 100)}
            {jsonString.length > 100 ? "..." : ""}
          </span>
          <button
            className="opacity-0 group-hover:opacity-100 p-0.5 hover:text-primary transition-opacity shrink-0"
            onClick={() => {
              setViewMode("tree");
            }}
            type="button"
          >
            <ListTree size={10} />
          </button>
        </div>
      )}
    </div>
  );
};
