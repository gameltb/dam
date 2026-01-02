import React from "react";
import { IconButton } from "../base/IconButton";
import {
  ZoomIn,
  ZoomOut,
  RotateCw,
  Maximize,
  Minimize,
  X,
  Focus,
} from "lucide-react";

interface MediaPreviewToolbarProps {
  label: string;
  activeIndex: number;
  totalItems: number;
  isImage: boolean;
  isVideo: boolean;
  videoMode: "fit" | "original";
  onZoomIn: (e?: React.MouseEvent) => void;
  onZoomOut: (e?: React.MouseEvent) => void;
  onRotate: (e?: React.MouseEvent) => void;
  onReset: () => void;
  onSetVideoMode: (mode: "fit" | "original") => void;
  onClose: (e: React.MouseEvent) => void;
}

export const MediaPreviewToolbar: React.FC<MediaPreviewToolbarProps> = ({
  label,
  activeIndex,
  totalItems,
  isImage,
  isVideo,
  videoMode,
  onZoomIn,
  onZoomOut,
  onRotate,
  onReset,
  onSetVideoMode,
  onClose,
}) => {
  return (
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
        background: "linear-gradient(to bottom, rgba(0,0,0,0.8), transparent)",
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
            {label}
          </span>
          <span style={{ fontSize: "12px", opacity: 0.6 }}>
            {activeIndex + 1} / {totalItems}
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
                onClick={onZoomIn}
                icon={<ZoomIn size={18} />}
                label="Zoom In"
              />
              <IconButton
                onClick={onZoomOut}
                icon={<ZoomOut size={18} />}
                label="Zoom Out"
              />
              <IconButton
                onClick={onRotate}
                icon={<RotateCw size={18} />}
                label="Rotate"
              />
              <IconButton
                onClick={onReset}
                icon={<Focus size={18} />}
                label="Reset View"
              />
            </>
          )}
          {isVideo && (
            <>
              <IconButton
                onClick={() => {
                  onSetVideoMode("fit");
                }}
                active={videoMode === "fit"}
                icon={<Minimize size={18} />}
                label="Fit to View"
              />
              <IconButton
                onClick={() => {
                  onSetVideoMode("original");
                }}
                active={videoMode === "original"}
                icon={<Maximize size={18} />}
                label="Original Size"
              />
            </>
          )}
        </div>
      </div>

      <IconButton
        onClick={onClose}
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
  );
};
