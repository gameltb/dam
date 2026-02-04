import { ChevronLeft, ChevronRight } from "lucide-react";
import React, { useCallback, useEffect, useMemo, useState } from "react";

import { MediaType } from "@/generated/flowcraft/v1/core/base_pb";
import { useMediaTransform } from "@/hooks/ux/useMediaTransform";
import { cn } from "@/lib/utils";
import { type AppNode, AppNodeType, VideoMode } from "@/types";

import { MediaContentRenderer } from "./MediaContentRenderer";
import { MediaPreviewToolbar } from "./MediaPreviewToolbar";

interface MediaPreviewProps {
  initialIndex: number;
  node: AppNode;
  onClose: () => void;
}

export const MediaPreview: React.FC<MediaPreviewProps> = ({ initialIndex, node, onClose }) => {
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

  const [videoMode, setVideoMode] = useState<VideoMode>(VideoMode.FIT);

  const media = node.type === AppNodeType.DYNAMIC ? node.data.media : null;
  const items = useMemo(() => {
    if (!media) return [];
    return [media.url, ...(media.galleryUrls ?? [])].filter(Boolean);
  }, [media]);

  const currentUrl = items[activeIndex];

  useEffect(() => {
    const preload = (url: string) => {
      if (!url) return;
      const img = new Image();
      img.src = url;
    };

    const nextUrl = activeIndex < items.length - 1 ? items[activeIndex + 1] : undefined;
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
      className="fixed inset-0 z-[5000] flex flex-col items-center justify-center bg-black/95 backdrop-blur-3xl animate-in fade-in duration-300"
      onMouseLeave={handleMouseUp}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
    >
      <MediaPreviewToolbar
        activeIndex={activeIndex}
        isImage={isImage}
        isVideo={isVideo}
        label={node.data.displayName ?? "Untitled Node"}
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
        className="relative flex-1 w-full flex items-center justify-center overflow-hidden"
        onWheel={isImage ? handleWheel : undefined}
      >
        {isLoading && (
          <div className="absolute z-20 flex flex-col items-center gap-4">
            <div className="w-12 h-12 rounded-full border-4 border-primary/20 border-t-primary animate-spin" />
          </div>
        )}

        {/* Navigation - Left */}
        {items.length > 1 && (
          <button
            className={cn(
              "absolute left-8 z-10 p-3 rounded-full bg-white/5 border border-white/10 text-white backdrop-blur-md transition-all hover:bg-white/10 hover:scale-110 active:scale-95 disabled:opacity-0 disabled:pointer-events-none",
              activeIndex === 0 && "opacity-0",
            )}
            disabled={activeIndex === 0}
            onClick={(e) => {
              e.stopPropagation();
              handlePrev();
            }}
          >
            <ChevronLeft size={32} />
          </button>
        )}

        <MediaContentRenderer
          currentUrl={currentUrl || ""}
          handleMouseDown={handleMouseDown}
          isDragging={isDragging}
          isImage={isImage}
          isVideo={isVideo}
          offset={offset}
          onLoaded={() => {
            setIsLoading(false);
          }}
          rotation={rotation}
          videoMode={videoMode}
          zoom={zoom}
        />

        {/* Navigation - Right */}
        {items.length > 1 && (
          <button
            className={cn(
              "absolute right-8 z-10 p-3 rounded-full bg-white/5 border border-white/10 text-white backdrop-blur-md transition-all hover:bg-white/10 hover:scale-110 active:scale-95 disabled:opacity-0 disabled:pointer-events-none",
              activeIndex === items.length - 1 && "opacity-0",
            )}
            disabled={activeIndex === items.length - 1}
            onClick={(e) => {
              e.stopPropagation();
              handleNext();
            }}
          >
            <ChevronRight size={32} />
          </button>
        )}
      </div>

      {/* Helper Footer */}
      <div className="z-10 w-full py-6 text-center text-[11px] font-medium uppercase tracking-[0.2em] text-white/30">
        {isImage
          ? "Scroll to Zoom • Drag to Move • Arrow keys to Switch • ESC to Close"
          : "Arrow keys to Switch • ESC to Close"}
      </div>
    </div>
  );
};
