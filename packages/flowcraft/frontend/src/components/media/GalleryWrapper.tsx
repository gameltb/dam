import { Layers, X } from "lucide-react";
import React, { useEffect, useRef, useState } from "react";

import { useFlowStore } from "@/store/flowStore";
import { FlowEvent, type MediaType } from "@/types";
import { IconButton } from "../base/IconButton";

interface GalleryWrapperProps {
  gallery: string[];
  id: string;
  mainContent: React.ReactNode;
  mediaType: MediaType;
  nodeHeight: number;
  nodeWidth: number;
  onExpand?: (expanded: boolean) => void;
  onGalleryItemContext?: (
    nodeId: string,
    url: string,
    mediaType: MediaType,
    x: number,
    y: number,
  ) => void;
  renderItem: (url: string) => React.ReactNode;
}

export const GalleryWrapper: React.FC<GalleryWrapperProps> = ({
  gallery,
  id,
  mainContent,
  mediaType,
  nodeHeight,
  nodeWidth,
  onExpand,
  onGalleryItemContext,
  renderItem,
}) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const lastNodeEvent = useFlowStore((state) => state.lastNodeEvent);
  const lastProcessedTimestamp = useRef(0);

  useEffect(() => {
    if (!isExpanded || !lastNodeEvent) return;

    // Only collapse if it's a NEW pane-click event that happened after we opened
    if (
      lastNodeEvent.type === FlowEvent.PANE_CLICK &&
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
        .onNodesChange([{ id, selected: true, type: "select" }]);
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
    useFlowStore
      .getState()
      .dispatchNodeEvent(FlowEvent.OPEN_PREVIEW, { index, nodeId: id });
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
        borderRadius: "inherit", // Inherit from BaseNode
        height: "100%",
        overflow: "visible", // Changed from dynamic to always visible
        pointerEvents: "none",
        position: "relative",
        width: "100%",
      }}
    >
      {/* Wrapper for main content to enforce clipping even when gallery is expanded */}
      <div
        style={{
          borderRadius: "inherit",
          height: "100%",
          left: 0,
          overflow: "hidden",
          pointerEvents: "none",
          position: "absolute",
          top: 0,
          width: "100%",
        }}
      >
        {mainContent}
      </div>

      {hasGallery && (
        <IconButton
          icon={isExpanded ? <X size={14} /> : <Layers size={14} />}
          label={
            isExpanded
              ? "Collapse Gallery"
              : `Expand Gallery (${String(gallery.length)})`
          }
          onClick={handleToggleExpand}
          style={{
            backgroundColor: isExpanded
              ? "rgba(255, 59, 48, 0.8)"
              : "rgba(0,0,0,0.6)",
            borderRadius: "12px",
            fontSize: "10px",
            height: "24px",
            padding: "0 8px",
            pointerEvents: "auto",
            position: "absolute",
            right: "5px",
            top: "5px",
            width: "auto",
            zIndex: 110,
          }}
        />
      )}

      {isExpanded && (
        <div
          style={{
            bottom: 0,
            display: "flex",
            flexDirection: "column-reverse",
            gap: "15px",
            left: 0,
            pointerEvents: "auto",
            position: "absolute",
            width: "max-content",
            zIndex: 100,
          }}
        >
          {galleryRows.map((rowItems, rowIndex) => (
            <div key={rowIndex} style={{ display: "flex", gap: "15px" }}>
              {rowIndex === 0 && (
                <div
                  style={{
                    height: nodeHeight,
                    visibility: "hidden",
                    width: nodeWidth,
                  }}
                />
              )}
              {rowItems.map((url, imgIndex) => (
                <div
                  className="nodrag"
                  key={imgIndex}
                  onClick={(e) => {
                    e.stopPropagation();
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
                  onDoubleClick={(e) => {
                    e.stopPropagation();
                    handleOpenPreview(gallery.indexOf(url) + 1);
                  }}
                  onMouseEnter={(e) =>
                    (e.currentTarget.style.transform = "scale(1.02)")
                  }
                  onMouseLeave={(e) =>
                    (e.currentTarget.style.transform = "scale(1)")
                  }
                  style={{
                    backdropFilter: "blur(10px)",
                    backgroundColor: "rgba(0,0,0,0.2)",
                    border: "1px solid rgba(255,255,255,0.2)",
                    borderRadius: "5px",
                    boxShadow: "0 8px 24px rgba(0,0,0,0.3)",
                    boxSizing: "border-box",
                    cursor: "pointer",
                    height: `${String(nodeHeight)}px`,
                    overflow: "hidden",
                    transition: "transform 0.2s",
                    width: `${String(nodeWidth)}px`,
                  }}
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
