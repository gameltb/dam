import { ChevronDown, ChevronRight, Copy } from "lucide-react";
import React, { useCallback, useState } from "react";
import { toast } from "react-hot-toast";

interface JsonTreeViewProps {
  data: any;
  depth?: number;
  isRoot?: boolean;
  name?: string;
}

export const JsonTreeView: React.FC<JsonTreeViewProps> = ({ data, depth = 0, isRoot = true, name }) => {
  const [isExpanded, setIsExpanded] = useState(depth < 2);

  const toggle = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      setIsExpanded(!isExpanded);
    },
    [isExpanded],
  );

  const copyToClipboard = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      navigator.clipboard.writeText(JSON.stringify(data, (_, v) => (typeof v === "bigint" ? v.toString() : v), 2));
      toast.success("JSON copied to clipboard", { id: "json-copy" });
    },
    [data],
  );

  const isObject = data !== null && typeof data === "object";
  const isArray = Array.isArray(data);

  if (!isObject) {
    return (
      <div className="flex items-center gap-2 group">
        {name && <span className="text-blue-400 font-bold">{name}:</span>}
        <span className={getPrimitiveColor(data)}>{formatPrimitive(data)}</span>
      </div>
    );
  }

  const keys = Object.keys(data);
  const isEmpty = keys.length === 0;

  return (
    <div className={isRoot ? "font-mono text-[11px]" : "ml-4"}>
      <div
        className="flex items-center gap-1 cursor-pointer hover:bg-muted/30 rounded px-1 transition-colors group relative"
        onClick={toggle}
      >
        {!isEmpty && (isExpanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />)}
        {name && <span className="text-blue-400 font-bold">{name}:</span>}
        <span className="text-muted-foreground">{isArray ? `Array[${keys.length}]` : `Object{${keys.length}}`}</span>

        {isRoot && (
          <button
            className="opacity-0 group-hover:opacity-100 p-1 hover:text-primary transition-opacity ml-2"
            onClick={copyToClipboard}
            title="Copy JSON"
          >
            <Copy size={10} />
          </button>
        )}
      </div>

      {isExpanded && !isEmpty && (
        <div className="border-l border-border/50 ml-1.5 pl-2 mt-0.5 space-y-0.5">
          {keys.map((key) => (
            <JsonTreeView data={data[key]} depth={depth + 1} isRoot={false} key={key} name={key} />
          ))}
        </div>
      )}
    </div>
  );
};

function formatPrimitive(val: any): string {
  if (val === null) return "null";
  if (val === undefined) return "undefined";
  if (typeof val === "string") return `"${val}"`;
  return String(val);
}

function getPrimitiveColor(val: any): string {
  if (val === null || val === undefined) return "text-muted-foreground/50";
  if (typeof val === "number" || typeof val === "bigint") return "text-orange-400";
  if (typeof val === "boolean") return "text-purple-400";
  if (typeof val === "string") return "text-green-400";
  return "text-foreground";
}
