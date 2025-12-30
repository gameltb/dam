import React, { useState, useEffect, useMemo, useCallback } from "react";
import { MediaType } from "../../generated/core/node_pb";
import { type AppNode } from "../../types";
import { IconButton } from "../base/IconButton";
import {
  ZoomIn,
  ZoomOut,
  RotateCw,
  Maximize,
  Minimize,
  X,
  ChevronLeft,
  ChevronRight,
  Focus,
} from "lucide-react";

interface MediaPreviewProps {
  node: AppNode;
  initialIndex: number;
  onClose: () => void;
}

export const MediaPreview: React.FC<MediaPreviewProps> = ({
  node,
  initialIndex,
  onClose,
}) => {
  const [activeIndex, setActiveIndex] = useState(initialIndex);
  const [isLoading, setIsLoading] = useState(false);

  // Image Transformations
  const [zoom, setZoom] = useState(1);
  const [rotation, setRotate] = useState(0);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  // Video Modes
  const [videoMode, setVideoMode] = useState<"fit" | "original">("fit");

  const media = node.type === "dynamic" ? node.data.media : null;
  const items = useMemo(() => {
    if (!media) return [];
    return [media.url, ...(media.galleryUrls ?? [])].filter(
      Boolean,
    ) as string[];
  }, [media]);

  const currentUrl = items[activeIndex];

  const resetTransform = useCallback(() => {
    setZoom(1);
    setRotate(0);
    setOffset({ x: 0, y: 0 });
  }, []);

  useEffect(() => {
    resetTransform();
  }, [activeIndex, resetTransform]);

  const handleZoomIn = (e?: React.MouseEvent) => {
    e?.stopPropagation();
    setZoom((prev) => Math.min(prev + 0.25, 5));
  };

  const handleZoomOut = (e?: React.MouseEvent) => {
    e?.stopPropagation();
    setZoom((prev) => Math.max(prev - 0.25, 0.5));
  };

  const handleRotate = (e?: React.MouseEvent) => {
    e?.stopPropagation();
    setRotate((prev) => prev + 90);
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (zoom <= 1) return;
    e.preventDefault();
    setIsDragging(true);
    setDragStart({ x: e.clientX - offset.x, y: e.clientY - offset.y });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;
    setOffset({
      x: e.clientX - dragStart.x,
      y: e.clientY - dragStart.y,
    });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleWheel = (e: React.WheelEvent) => {
    if (e.deltaY < 0) handleZoomIn();
    else handleZoomOut();
  };

  // Preloading Logic
  useEffect(() => {
    const preload = (url: string) => {
      if (!url) return;
      const img = new Image();
      img.src = url;
    };

    const nextUrl =
      activeIndex < items.length - 1 ? items[activeIndex + 1] : undefined;
    const prevUrl = activeIndex > 0 ? items[activeIndex - 1] : undefined;

    if (nextUrl) preload(nextUrl);
    if (prevUrl) preload(prevUrl);
  }, [activeIndex, items]);

  const handleSwitch = useCallback(
    (newIndex: number) => {
      const url = items[newIndex];
      if (!url) return;

      if (media?.type === MediaType.MEDIA_IMAGE) {
        const img = new Image();
        img.src = url;
        setIsLoading(!img.complete);
      } else {
        setIsLoading(true);
      }
      setActiveIndex(newIndex);
    },
    [items, media?.type],
  );

  const handlePrev = useCallback(() => {
    if (activeIndex > 0) handleSwitch(activeIndex - 1);
  }, [activeIndex, handleSwitch]);

  const handleNext = useCallback(() => {
    if (activeIndex < items.length - 1) handleSwitch(activeIndex + 1);
  }, [activeIndex, items.length, handleSwitch]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      if (e.key === "ArrowLeft") handlePrev();
      if (e.key === "ArrowRight") handleNext();
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [handlePrev, handleNext, onClose]);

  if (!media || items.length === 0) return null;

  const isImage = media.type === MediaType.MEDIA_IMAGE;
  const isVideo = media.type === MediaType.MEDIA_VIDEO;

  return (
    <div
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        width: "100vw",
        height: "100vh",
        backgroundColor: "rgba(0,0,0,0.95)",
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        alignItems: "center",
        zIndex: 5000,
        backdropFilter: "blur(25px)",
      }}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    >
      {/* Header / Toolbar */}
      <div
        style={{
          position: "absolute",
          top: 0,
          width: "100%",
          padding: "20px 40px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          color: "white",
          background:
            "linear-gradient(to bottom, rgba(0,0,0,0.8), transparent)",
          boxSizing: "border-box",
          zIndex: 100,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "20px" }}>
          <div style={{ display: "flex", flexDirection: "column" }}>
            <span
              style={{
                fontSize: "18px",
                fontWeight: 600,
                letterSpacing: "-0.5px",
              }}
            >
              {node.data.label}
            </span>
            <span style={{ fontSize: "12px", opacity: 0.6 }}>
              {activeIndex + 1} / {items.length}
            </span>
          </div>

          <div
            style={{
              display: "flex",
              backgroundColor: "rgba(255,255,255,0.05)",
              borderRadius: "10px",
              padding: "4px",
              gap: "4px",
              backdropFilter: "blur(10px)",
              border: "1px solid rgba(255,255,255,0.1)",
            }}
          >
            {isImage && (
              <>
                <IconButton
                  onClick={handleZoomIn}
                  icon={<ZoomIn size={18} />}
                  label="Zoom In"
                />
                <IconButton
                  onClick={handleZoomOut}
                  icon={<ZoomOut size={18} />}
                  label="Zoom Out"
                />
                <IconButton
                  onClick={handleRotate}
                  icon={<RotateCw size={18} />}
                  label="Rotate"
                />
                <IconButton
                  onClick={resetTransform}
                  icon={<Focus size={18} />}
                  label="Reset View"
                />
              </>
            )}
            {isVideo && (
              <>
                <IconButton
                  onClick={() => setVideoMode("fit")}
                  active={videoMode === "fit"}
                  icon={<Minimize size={18} />}
                  label="Fit to View"
                />
                <IconButton
                  onClick={() => setVideoMode("original")}
                  active={videoMode === "original"}
                  icon={<Maximize size={18} />}
                  label="Original Size"
                />
              </>
            )}
          </div>
        </div>

        <IconButton
          onClick={(e) => {
            e.stopPropagation();
            onClose();
          }}
          icon={<X size={20} />}
          label="Close"
          style={{
            width: "40px",
            height: "40px",
            borderRadius: "12px",
            backgroundColor: "rgba(255, 59, 48, 0.15)",
            borderColor: "rgba(255, 59, 48, 0.2)",
            color: "#ff3b30",
          }}
        />
      </div>

      {/* Main Content Area */}
      <div
        style={{
          flex: 1,
          width: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          position: "relative",
          overflow: "hidden",
        }}
        onWheel={isImage ? handleWheel : undefined}
      >
        {isLoading && (
          <div
            style={{
              position: "absolute",
              zIndex: 20,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: "15px",
            }}
          >
            <div
              className="preview-spinner"
              style={{
                width: "40px",
                height: "40px",
                border: "3px solid rgba(255,255,255,0.1)",
                borderTop: "3px solid #646cff",
                borderRadius: "50%",
                animation: "spin 1s linear infinite",
              }}
            />
          </div>
        )}

        {/* Navigation - Left */}
        {items.length > 1 && (
          <IconButton
            disabled={activeIndex === 0}
            onClick={(e) => {
              e.stopPropagation();
              handlePrev();
            }}
            icon={<ChevronLeft size={32} />}
            style={{
              ...navButtonStyle,
              left: "40px",
              opacity: activeIndex === 0 ? 0 : 1,
            }}
          />
        )}

        <div
          onMouseDown={isImage ? handleMouseDown : undefined}
          style={{
            maxWidth: "100%",
            maxHeight: "100%",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            cursor: zoom > 1 ? (isDragging ? "grabbing" : "grab") : "default",
            transition: isDragging ? "none" : "transform 0.2s ease-out",
            transform: `translate(${offset.x}px, ${offset.y}px) scale(${zoom}) rotate(${rotation}deg)`,
          }}
        >
          {isImage ? (
            <img
              src={currentUrl}
              onLoad={() => setIsLoading(false)}
              style={{
                maxWidth: "90vw",
                maxHeight: "85vh",
                objectFit: "contain",
                boxShadow: "0 30px 60px rgba(0,0,0,0.8)",
                pointerEvents: "none",
              }}
              alt="Preview"
            />
          ) : isVideo ? (
            <video
              key={currentUrl}
              src={currentUrl}
              onLoadedData={() => setIsLoading(false)}
              controls
              autoPlay
              style={{
                width: videoMode === "original" ? "auto" : "90vw",
                height: videoMode === "original" ? "auto" : "85vh",
                maxWidth: "100%",
                maxHeight: "100%",
                objectFit: videoMode === "original" ? "none" : "contain",
                boxShadow: "0 30px 60px rgba(0,0,0,0.8)",
              }}
            />
          ) : (
            <div
              style={{
                width: "400px",
                padding: "40px",
                backgroundColor: "#1a1a1a",
                borderRadius: "20px",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                gap: "30px",
                border: "1px solid rgba(255,255,255,0.1)",
              }}
            >
              <audio
                key={currentUrl}
                src={currentUrl}
                onLoadedData={() => setIsLoading(false)}
                controls
                autoPlay
                style={{ width: "100%" }}
              />
            </div>
          )}
        </div>

        {/* Navigation - Right */}
        {items.length > 1 && (
          <IconButton
            disabled={activeIndex === items.length - 1}
            onClick={(e) => {
              e.stopPropagation();
              handleNext();
            }}
            icon={<ChevronRight size={32} />}
            style={{
              ...navButtonStyle,
              right: "40px",
              opacity: activeIndex === items.length - 1 ? 0 : 1,
            }}
          />
        )}
      </div>

      <div
        style={{
          padding: "20px",
          color: "rgba(255,255,255,0.4)",
          fontSize: "13px",
          width: "100%",
          textAlign: "center",
          zIndex: 10,
        }}
      >
        {isImage
          ? "Scroll to Zoom • Drag to Move • Arrow keys to Switch • ESC to Close"
          : "Arrow keys to Switch • ESC to Close"}
      </div>
    </div>
  );
};

const toolbarButtonStyle: React.CSSProperties = {
  background: "none",
  border: "none",
  color: "white",
  width: "32px",
  height: "32px",
  borderRadius: "6px",
  cursor: "pointer",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  fontSize: "14px",
  transition: "background 0.2s",
};

const navButtonStyle: React.CSSProperties = {
  position: "absolute",
  width: "50px",
  height: "50px",
  background: "rgba(255,255,255,0.05)",
  color: "white",
  border: "1px solid rgba(255,255,255,0.1)",
  borderRadius: "50%",
  cursor: "pointer",
  fontSize: "24px",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  backdropFilter: "blur(10px)",
  zIndex: 10,
  transition: "all 0.2s",
};

