import React from "react";

import { cn } from "@/lib/utils";

interface NodeInfoPanelProps {
  height: number;
  nodeId: string;
  templateId: string;
  width: number;
  x: number;
  y: number;
}

export const NodeInfoPanel: React.FC<NodeInfoPanelProps> = ({
  height,
  nodeId,
  templateId,
  width,
  x,
  y,
}) => (
  <div
    className={cn(
      "nodrag nopan absolute -top-12 left-1/2 -translate-x-1/2",
      "bg-black/95 backdrop-blur-md border border-primary-color rounded-lg px-3.5 py-1.5",
      "flex gap-3.5 items-center shadow-2xl z-[9999] pointer-events-none whitespace-nowrap",
      "text-[11px] font-semibold text-white animate-node-info-fade-in",
    )}
  >
    <div className="flex items-center gap-1.5">
      <span className="opacity-50 text-[9px]">NODE_ID</span>
      <span>{nodeId.slice(0, 8)}</span>
    </div>

    <div className="w-px h-3 bg-white/15" />

    <div className="flex items-center gap-1.5">
      <span className="opacity-50 text-[9px]">TEMPLATE_ID</span>
      <span className="text-primary-color tracking-wider uppercase">
        {templateId}
      </span>
    </div>

    <div className="w-px h-3 bg-white/15" />

    <div className="flex items-center gap-3">
      <div className="flex items-center gap-1">
        <span className="opacity-50 text-[9px]">POS</span>
        <span className="text-green-500">X:</span>
        <span className="min-w-[20px]">{Math.round(x)}</span>
        <span className="text-green-500 ml-1">Y:</span>
        <span className="min-w-[20px]">{Math.round(y)}</span>
      </div>

      <div className="w-px h-2.5 bg-white/10" />

      <div className="flex items-center gap-1">
        <span className="opacity-50 text-[9px]">SIZE</span>
        <span>
          {Math.round(width)} × {Math.round(height)}
        </span>
      </div>
    </div>
  </div>
);
