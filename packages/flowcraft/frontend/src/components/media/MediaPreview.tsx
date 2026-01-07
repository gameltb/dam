import { ChevronLeft, ChevronRight } from "lucide-react";
import React, { useCallback, useEffect, useMemo, useState } from "react";

import { MediaType } from "../../generated/flowcraft/v1/core/node_pb";
import { useMediaTransform } from "../../hooks/useMediaTransform";
import { type AppNode, AppNodeType } from "../../types";
import { IconButton } from "../base/IconButton";
import { MediaPreviewToolbar } from "./MediaPreviewToolbar";

interface MediaPreviewProps {
  initialIndex: number;
  node: AppNode;
  onClose: () => void;
}

export const MediaPreview: React.FC<MediaPreviewProps> = ({
  initialIndex,
  node,
  onClose,
}) => {
  const [activeIndex, setActiveIndex] = useState(initialIndex);
  const [isLoading, setIsLoading] = useState(false);

  const {
    handleMouseDown,
    handleMouseMove,
    handleMouseUp,
    handleRotate,
    handleWheel,
    handleZoomIn,
    handleZoomOut,
    isDragging,
    offset,
    resetTransform,
    rotation,
    zoom,
  } = useMediaTransform(activeIndex);

  // Video Modes
  const [videoMode, setVideoMode] = useState<"fit" | "original">("fit");

  const media = node.type === AppNodeType.DYNAMIC ? node.data.media : null;
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
      onMouseLeave={handleMouseUp}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      style={{
        alignItems: "center",
        backdropFilter: "blur(25px)",
        backgroundColor: "rgba(0,0,0,0.95)",
        display: "flex",
        flexDirection: "column",
        height: "100vh",
        justifyContent: "center",
        left: 0,
        position: "fixed",
        top: 0,
        width: "100vw",
        zIndex: 5000,
      }}
    >
      <MediaPreviewToolbar
        activeIndex={activeIndex}
        isImage={isImage}
        isVideo={isVideo}
        label={node.data.label ?? "Untitled Node"}
        onClose={(e) => {
          e.stopPropagation();
          onClose();
        }}
        onReset={resetTransform}
        onRotate={handleRotate}
        onSetVideoMode={setVideoMode}
        onZoomIn={handleZoomIn}
        onZoomOut={handleZoomOut}
        totalItems={items.length}
        videoMode={videoMode}
      />

      {/* Main Content Area */}
      <div
        onWheel={isImage ? handleWheel : undefined}
        style={{
          alignItems: "center",
          display: "flex",
          flex: 1,
          justifyContent: "center",
          overflow: "hidden",
          position: "relative",
          width: "100%",
        }}
      >
        {isLoading && (
          <div
            style={{
              alignItems: "center",
              display: "flex",
              flexDirection: "column",
              gap: "15px",
              position: "absolute",
              zIndex: 20,
            }}
          >
            <div
              className="preview-spinner"
              style={{
                animation: "spin 1s linear infinite",
                border: "3px solid rgba(255,255,255,0.1)",
                borderRadius: "50%",
                borderTop: "3px solid #646cff",
                height: "40px",
                width: "40px",
              }}
            />
          </div>
        )}

        {/* Navigation - Left */}
        {items.length > 1 && (
          <IconButton
            disabled={activeIndex === 0}
            icon={<ChevronLeft size={32} />}
            onClick={(e) => {
              e.stopPropagation();
              handlePrev();
            }}
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
            alignItems: "center",
            cursor: zoom > 1 ? (isDragging ? "grabbing" : "grab") : "default",
            display: "flex",
            justifyContent: "center",
            maxHeight: "100%",
            maxWidth: "100%",
            transform: `translate(${String(offset.x)}px, ${String(offset.y)}px) scale(${String(zoom)}) rotate(${String(rotation)}deg)`,
            transition: isDragging ? "none" : "transform 0.2s ease-out",
          }}
        >
          {isImage ? (
            <img
              alt="Preview"
              onLoad={() => {
                setIsLoading(false);
              }}
              src={currentUrl}
              style={{
                boxShadow: "0 30px 60px rgba(0,0,0,0.8)",
                maxHeight: "85vh",
                maxWidth: "90vw",
                objectFit: "contain",
                pointerEvents: "none",
              }}
            />
          ) : isVideo ? (
            <video
              autoPlay
              controls
              key={currentUrl}
              onLoadedData={() => {
                setIsLoading(false);
              }}
              src={currentUrl}
              style={{
                boxShadow: "0 30px 60px rgba(0,0,0,0.8)",
                height: videoMode === "original" ? "auto" : "85vh",
                maxHeight: "100%",
                maxWidth: "100%",
                objectFit: videoMode === "original" ? "none" : "contain",
                width: videoMode === "original" ? "auto" : "90vw",
              }}
            />
          ) : (
            <div
              style={{
                alignItems: "center",
                backgroundColor: "#1a1a1a",
                border: "1px solid rgba(255,255,255,0.1)",
                borderRadius: "20px",
                display: "flex",
                flexDirection: "column",
                gap: "30px",
                padding: "40px",
                width: "400px",
              }}
            >
              <audio
                autoPlay
                controls
                key={currentUrl}
                onLoadedData={() => {
                  setIsLoading(false);
                }}
                src={currentUrl}
                style={{ width: "100%" }}
              />
            </div>
          )}
        </div>

        {/* Navigation - Right */}
        {items.length > 1 && (
          <IconButton
            disabled={activeIndex === items.length - 1}
            icon={<ChevronRight size={32} />}
            onClick={(e) => {
              e.stopPropagation();
              handleNext();
            }}
            style={{
              ...navButtonStyle,
              opacity: activeIndex === items.length - 1 ? 0 : 1,
              right: "40px",
            }}
          />
        )}
      </div>

      <div
        style={{
          color: "rgba(255,255,255,0.4)",
          fontSize: "13px",
          padding: "20px",
          textAlign: "center",
          width: "100%",
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
  alignItems: "center",
  backdropFilter: "blur(10px)",
  background: "rgba(255,255,255,0.05)",
  border: "1px solid rgba(255,255,255,0.1)",
  borderRadius: "50%",
  color: "white",
  cursor: "pointer",
  display: "flex",
  fontSize: "24px",
  height: "50px",
  justifyContent: "center",
  position: "absolute",
  transition: "all 0.2s",
  width: "50px",
  zIndex: 10,
};
