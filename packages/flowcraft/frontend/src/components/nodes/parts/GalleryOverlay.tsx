import { Layers, Maximize2, X } from "lucide-react";
import React, { memo, useCallback } from "react";

import { useUiProperty } from "@/hooks/core/useUiProperty";
import { cn } from "@/lib/utils";
import { useFlowStore } from "@/store/flowStore";
import { FlowEvent, type MediaType } from "@/types";

interface GalleryOverlayProps {
  gallery: string[];
  id: string;
  mediaType: MediaType;
  nodeHeight: number;
  nodeWidth: number;
  onGalleryItemContext?: (nodeId: string, url: string, mediaType: MediaType, x: number, y: number) => void;
  renderItem: (url: string) => React.ReactNode;
}

/**
 * GalleryOverlay
 * A specialized part for nodes containing multiple media items.
 * Synchronizes expansion state across clients via useUiProperty.
 */
export const GalleryOverlay: React.FC<GalleryOverlayProps> = memo(
  ({ gallery, id, mediaType, nodeHeight, nodeWidth, onGalleryItemContext, renderItem }) => {
    const [isExpanded, setIsExpanded] = useUiProperty(id, "isGalleryExpanded", false);
    const dispatchNodeEvent = useFlowStore((s) => s.dispatchNodeEvent);

    const handleToggleExpand = useCallback(
      (e: React.MouseEvent) => {
        e.stopPropagation();
        setIsExpanded(!isExpanded);
      },
      [isExpanded, setIsExpanded],
    );

    const handleOpenPreview = (index: number) => {
      dispatchNodeEvent(FlowEvent.OPEN_PREVIEW, { index, nodeId: id });
    };

    if (gallery.length === 0) return null;

    return (
      <>
        {/* Floating Toggle Button */}
        <button
          className={cn(
            "absolute top-2 right-2 z-[110] flex items-center gap-1.5 px-2 py-1 rounded-lg text-[10px] font-bold uppercase tracking-tight transition-all pointer-events-auto",
            isExpanded
              ? "bg-destructive text-destructive-foreground shadow-lg scale-100"
              : "bg-black/60 text-white backdrop-blur-md hover:bg-black/80 scale-90 hover:scale-100",
          )}
          onClick={handleToggleExpand}
        >
          {isExpanded ? <X size={12} /> : <Layers size={12} />}
          {isExpanded ? "Close" : `Gallery (${gallery.length})`}
        </button>

        {/* Expanded Grid View */}
        {isExpanded && (
          <div
            className="absolute top-0 left-full ml-4 z-[100] flex flex-wrap gap-4 p-4 bg-background/40 backdrop-blur-xl border border-border/50 rounded-2xl shadow-2xl animate-in fade-in slide-in-from-left-4 duration-300 pointer-events-auto"
            style={{ maxWidth: "80vw", width: "max-content" }}
          >
            {gallery.map((url, index) => (
              <div
                className="group/gallery-item relative rounded-xl border border-white/10 bg-black/20 overflow-hidden shadow-lg transition-all hover:scale-[1.02] active:scale-95 cursor-pointer"
                key={index}
                onContextMenu={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  onGalleryItemContext?.(id, url, mediaType, e.clientX, e.clientY);
                }}
                onDoubleClick={(e) => {
                  e.stopPropagation();
                  handleOpenPreview(index + 1); // Main is index 0
                }}
                style={{ height: nodeHeight, width: nodeWidth }}
              >
                <div className="w-full h-full pointer-events-none">{renderItem(url)}</div>

                {/* Hover Actions */}
                <div className="absolute inset-0 bg-black/40 opacity-0 group-hover/gallery-item:opacity-100 transition-opacity flex items-center justify-center">
                  <Maximize2 className="text-white" size={24} />
                </div>
              </div>
            ))}
          </div>
        )}
      </>
    );
  },
);

GalleryOverlay.displayName = "GalleryOverlay";
