import React, { useState, useEffect, useMemo, useCallback } from "react";
import { MediaType } from "../../generated/core/node_pb";
import { type AppNode } from "../../types";
import { IconButton } from "../base/IconButton";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { useMediaTransform } from "../../hooks/useMediaTransform";
import { MediaPreviewToolbar } from "./MediaPreviewToolbar";

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

  const {
    zoom,
    rotation,
    offset,
    isDragging,
    resetTransform,
    handleZoomIn,
    handleZoomOut,
    handleRotate,
    handleMouseDown,
    handleMouseMove,
    handleMouseUp,
    handleWheel,
  } = useMediaTransform(activeIndex);

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
      <MediaPreviewToolbar
        label={node.data.label ?? "Untitled Node"}
        activeIndex={activeIndex}
        totalItems={items.length}
        isImage={isImage}
        isVideo={isVideo}
        videoMode={videoMode}
        onZoomIn={handleZoomIn}
        onZoomOut={handleZoomOut}
        onRotate={handleRotate}
        onReset={resetTransform}
        onSetVideoMode={setVideoMode}
        onClose={(e) => {
          e.stopPropagation();
          onClose();
        }}
      />

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
            transform: `translate(${String(offset.x)}px, ${String(offset.y)}px) scale(${String(zoom)}) rotate(${String(rotation)}deg)`,
          }}
        >
          {isImage ? (
            <img
              src={currentUrl}
              onLoad={() => {
                setIsLoading(false);
              }}
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
              onLoadedData={() => {
                setIsLoading(false);
              }}
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
                onLoadedData={() => {
                  setIsLoading(false);
                }}
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
