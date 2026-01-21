import React from "react";

import { VideoMode } from "@/types";

interface MediaContentRendererProps {
  currentUrl: string;
  handleMouseDown?: (e: React.MouseEvent) => void;
  isDragging: boolean;
  isImage: boolean;
  isVideo: boolean;
  offset: { x: number; y: number };
  onLoaded?: () => void;
  rotation: number;
  videoMode: VideoMode;
  zoom: number;
}

export const MediaContentRenderer: React.FC<MediaContentRendererProps> = ({
  currentUrl,
  handleMouseDown,
  isDragging,
  isImage,
  isVideo,
  offset,
  onLoaded,
  rotation,
  videoMode,
  zoom,
}) => {
  return (
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
          onLoad={onLoaded}
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
          onLoadedData={onLoaded}
          src={currentUrl}
          style={{
            boxShadow: "0 30px 60px rgba(0,0,0,0.8)",
            height: videoMode === VideoMode.ORIGINAL ? "auto" : "85vh",
            maxHeight: "100%",
            maxWidth: "100%",
            objectFit: videoMode === VideoMode.ORIGINAL ? "none" : "contain",
            width: videoMode === VideoMode.ORIGINAL ? "auto" : "90vw",
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
            onLoadedData={onLoaded}
            src={currentUrl}
            style={{ width: "100%" }}
          />
        </div>
      )}
    </div>
  );
};
