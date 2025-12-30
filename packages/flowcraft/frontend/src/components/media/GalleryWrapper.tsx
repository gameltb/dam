import React, { useState, useEffect, useRef } from "react";
import { useFlowStore } from "../../store/flowStore";
import type { MediaType } from "../../types";
import { IconButton } from "../base/IconButton";
import { Layers, X } from "lucide-react";

interface GalleryWrapperProps {
  id: string;
  nodeWidth: number;
  nodeHeight: number;
  mainContent: React.ReactNode;
  gallery: string[];
  mediaType: MediaType;
  renderItem: (url: string) => React.ReactNode;
  onGalleryItemContext?: (
    nodeId: string,
    url: string,
    mediaType: MediaType,
    x: number,
    y: number,
  ) => void;
  onExpand?: (expanded: boolean) => void;
}

export const GalleryWrapper: React.FC<GalleryWrapperProps> = ({
  id,
  nodeWidth,
  nodeHeight,
  mainContent,
  gallery,
  mediaType,
  renderItem,
  onGalleryItemContext,
  onExpand,
}) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const lastNodeEvent = useFlowStore((state) => state.lastNodeEvent);
  const lastProcessedTimestamp = useRef(0);

  useEffect(() => {
    if (!isExpanded || !lastNodeEvent) return;

    // Only collapse if it's a NEW pane-click event that happened after we opened
    if (
      lastNodeEvent.type === "pane-click" &&
      lastNodeEvent.timestamp > lastProcessedTimestamp.current
    ) {
      setIsExpanded(false); // eslint-disable-line react-hooks/set-state-in-effect
      onExpand?.(false);
      lastProcessedTimestamp.current = lastNodeEvent.timestamp;
    }
  }, [lastNodeEvent, isExpanded, onExpand]);

  const hasGallery = gallery.length > 0;

  const handleToggleExpand = (e: React.MouseEvent) => {
    e.stopPropagation();

    // Optimization: If node is not selected, select it first
    const nodes = useFlowStore.getState().nodes;
    const currentNode = nodes.find((n) => n.id === id);
    if (currentNode && !currentNode.selected) {
      useFlowStore
        .getState()
        .onNodesChange([{ id, type: "select", selected: true }]);
    }

    const next = !isExpanded;

    // When opening, we mark the CURRENT event as processed so we only close on FUTURE events
    if (next && lastNodeEvent) {
      lastProcessedTimestamp.current = lastNodeEvent.timestamp;
    }

    setIsExpanded(next);
    onExpand?.(next);
  };

  const handleOpenPreview = (index: number) => {
    useFlowStore.getState().dispatchNodeEvent("open-preview", { nodeId: id, index });
  };

  const getGalleryRows = () => {
    const n = gallery.length;
    if (n === 0) return [];

    const rows: string[][] = [];
    const remaining = [...gallery];

    if (n <= 2) {
      rows.push(remaining);
    } else if (n <= 4) {
      rows.push(remaining.splice(0, 2));
      rows.push(remaining);
    } else if (n <= 6) {
      rows.push(remaining.splice(0, 3));
      rows.push(remaining);
    } else {
      const total = n + 1;
      const numRows = Math.round(Math.sqrt(total));
      const targetWidth = Math.ceil(total / numRows);
      rows.push(remaining.splice(0, targetWidth - 1));
      while (remaining.length > 0) {
        rows.push(remaining.splice(0, targetWidth));
      }
    }
    return rows;
  };

  const galleryRows = getGalleryRows();

  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        position: "relative",
        overflow: "visible", // Changed from dynamic to always visible
        borderRadius: "inherit", // Inherit from BaseNode
        pointerEvents: "none",
      }}
    >
      {/* Wrapper for main content to enforce clipping even when gallery is expanded */}
      <div
        style={{
          width: "100%",
          height: "100%",
          overflow: "hidden",
          borderRadius: "inherit",
          position: "absolute",
          top: 0,
          left: 0,
          pointerEvents: "none",
        }}
      >
        {mainContent}
      </div>

      {hasGallery && (
        <IconButton
          onClick={handleToggleExpand}
          icon={isExpanded ? <X size={14} /> : <Layers size={14} />}
          label={isExpanded ? "Collapse Gallery" : `Expand Gallery (${String(gallery.length)})`}
          style={{
            position: "absolute",
            top: "5px",
            right: "5px",
            zIndex: 110,
            width: "auto",
            height: "24px",
            padding: "0 8px",
            borderRadius: "12px",
            fontSize: "10px",
            backgroundColor: isExpanded ? "rgba(255, 59, 48, 0.8)" : "rgba(0,0,0,0.6)",
            pointerEvents: "auto",
          }}
        />
      )}

      {isExpanded && (
        <div
          style={{
            position: "absolute",
            left: 0,
            bottom: 0,
            display: "flex",
            flexDirection: "column-reverse",
            gap: "15px",
            zIndex: 100,
            width: "max-content",
            pointerEvents: "auto",
          }}
        >
          {galleryRows.map((rowItems, rowIndex) => (
            <div key={rowIndex} style={{ display: "flex", gap: "15px" }}>
              {rowIndex === 0 && (
                <div
                  style={{
                    width: nodeWidth,
                    height: nodeHeight,
                    visibility: "hidden",
                  }}
                />
              )}
              {rowItems.map((url, imgIndex) => (
                <div
                  key={imgIndex}
                  className="nodrag"
                  onClick={(e) => {
                    e.stopPropagation();
                  }}
                  onDoubleClick={(e) => {
                    e.stopPropagation();
                    handleOpenPreview(gallery.indexOf(url) + 1);
                  }}
                  onContextMenu={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    onGalleryItemContext?.(
                      id,
                      url,
                      mediaType,
                      e.clientX,
                      e.clientY,
                    );
                  }}
                  style={{
                    width: `${String(nodeWidth)}px`,
                    height: `${String(nodeHeight)}px`,
                    border: "1px solid rgba(255,255,255,0.2)",
                    borderRadius: "5px",
                    overflow: "hidden",
                    backgroundColor: "rgba(0,0,0,0.2)",
                    boxShadow: "0 8px 24px rgba(0,0,0,0.3)",
                    cursor: "pointer",
                    transition: "transform 0.2s",
                    backdropFilter: "blur(10px)",
                    boxSizing: "border-box",
                  }}
                  onMouseEnter={(e) =>
                    (e.currentTarget.style.transform = "scale(1.02)")
                  }
                  onMouseLeave={(e) =>
                    (e.currentTarget.style.transform = "scale(1)")
                  }
                >
                  {renderItem(url)}
                </div>
              ))}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
