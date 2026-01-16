import { Focus, Maximize, Minimize, RotateCw, X, ZoomIn, ZoomOut } from "lucide-react";
import React from "react";

import { VideoMode } from "@/types";

import { IconButton } from "../base/IconButton";

interface MediaPreviewToolbarProps {
  activeIndex: number;
  isImage: boolean;
  isVideo: boolean;
  label: string;
  onClose: (e: React.MouseEvent) => void;
  onReset: () => void;
  onRotate: (e?: React.MouseEvent) => void;
  onSetVideoMode: (mode: VideoMode) => void;
  onZoomIn: (e?: React.MouseEvent) => void;
  onZoomOut: (e?: React.MouseEvent) => void;
  totalItems: number;
  videoMode: VideoMode;
}

export const MediaPreviewToolbar: React.FC<MediaPreviewToolbarProps> = ({
  activeIndex,
  isImage,
  isVideo,
  label,
  onClose,
  onReset,
  onRotate,
  onSetVideoMode,
  onZoomIn,
  onZoomOut,
  totalItems,
  videoMode,
}) => {
  return (
    <div
      style={{
        alignItems: "center",
        background: "linear-gradient(to bottom, rgba(0,0,0,0.8), transparent)",
        boxSizing: "border-box",
        color: "white",
        display: "flex",
        justifyContent: "space-between",
        padding: "20px 40px",
        position: "absolute",
        top: 0,
        width: "100%",
        zIndex: 100,
      }}
    >
      <div style={{ alignItems: "center", display: "flex", gap: "20px" }}>
        <div style={{ display: "flex", flexDirection: "column" }}>
          <span
            style={{
              fontSize: "18px",
              fontWeight: 600,
              letterSpacing: "-0.5px",
            }}
          >
            {label}
          </span>
          <span style={{ fontSize: "12px", opacity: 0.6 }}>
            {activeIndex + 1} / {totalItems}
          </span>
        </div>

        <div
          style={{
            backdropFilter: "blur(10px)",
            backgroundColor: "rgba(255,255,255,0.05)",
            border: "1px solid rgba(255,255,255,0.1)",
            borderRadius: "10px",
            display: "flex",
            gap: "4px",
            padding: "4px",
          }}
        >
          {isImage && (
            <>
              <IconButton icon={<ZoomIn size={18} />} label="Zoom In" onClick={onZoomIn} />
              <IconButton icon={<ZoomOut size={18} />} label="Zoom Out" onClick={onZoomOut} />
              <IconButton icon={<RotateCw size={18} />} label="Rotate" onClick={onRotate} />
              <IconButton icon={<Focus size={18} />} label="Reset View" onClick={onReset} />
            </>
          )}
          {isVideo && (
            <>
              <IconButton
                active={videoMode === VideoMode.FIT}
                icon={<Minimize size={18} />}
                label="Fit to View"
                onClick={() => {
                  onSetVideoMode(VideoMode.FIT);
                }}
              />
              <IconButton
                active={videoMode === VideoMode.ORIGINAL}
                icon={<Maximize size={18} />}
                label="Original Size"
                onClick={() => {
                  onSetVideoMode(VideoMode.ORIGINAL);
                }}
              />
            </>
          )}
        </div>
      </div>

      <IconButton
        icon={<X size={20} />}
        label="Close"
        onClick={onClose}
        style={{
          backgroundColor: "rgba(255, 59, 48, 0.15)",
          borderColor: "rgba(255, 59, 48, 0.2)",
          borderRadius: "12px",
          color: "#ff3b30",
          height: "40px",
          width: "40px",
        }}
      />
    </div>
  );
};
