import React, { useState, useEffect } from "react";

interface GalleryWrapperProps {
  id: string;
  nodeWidth: number;
  nodeHeight: number;
  mainContent: React.ReactNode;
  gallery: string[];
  renderItem: (url: string) => React.ReactNode;
  onGalleryItemContext?: (
    nodeId: string,
    url: string,
    x: number,
    y: number,
  ) => void;
}

export const GalleryWrapper: React.FC<GalleryWrapperProps> = ({
  id,
  nodeWidth,
  nodeHeight,
  mainContent,
  gallery,
  renderItem,
  onGalleryItemContext,
}) => {
  const [isExpanded, setIsExpanded] = useState(false);

  useEffect(() => {
    if (!isExpanded) return;
    const handleClickOutside = () => setIsExpanded(false);
    window.addEventListener("click", handleClickOutside);
    return () => window.removeEventListener("click", handleClickOutside);
  }, [isExpanded]);

  const hasGallery = gallery.length > 0;

  const handleToggleExpand = (e: React.MouseEvent) => {
    e.stopPropagation();
    setIsExpanded(!isExpanded);
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
        overflow: isExpanded ? "visible" : "hidden",
        borderRadius: "inherit", // Inherit from BaseNode
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
        }}
      >
        {mainContent}
      </div>

      {hasGallery && (
        <div
          onClick={handleToggleExpand}
          style={{
            position: "absolute",
            top: "5px",
            right: "5px",
            backgroundColor: "rgba(0,0,0,0.6)",
            color: "white",
            borderRadius: "12px",
            padding: "2px 8px",
            fontSize: "10px",
            cursor: "pointer",
            zIndex: 10,
            backdropFilter: "blur(4px)",
            border: "1px solid rgba(255,255,255,0.2)",
          }}
        >
          +{gallery.length}
        </div>
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
                  onClick={(e) => e.stopPropagation()}
                  onContextMenu={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    onGalleryItemContext?.(id, url, e.clientX, e.clientY);
                  }}
                  style={{
                    width: `${nodeWidth}px`,
                    height: `${nodeHeight}px`,
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
