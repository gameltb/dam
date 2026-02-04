import { Focus, Maximize, Minimize, RotateCw, X, ZoomIn, ZoomOut } from "lucide-react";
import React from "react";

import { cn } from "@/lib/utils";
import { VideoMode } from "@/types";

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

const ToolbarButton: React.FC<{
  active?: boolean;
  icon: React.ReactNode;
  label: string;
  onClick: (e: React.MouseEvent) => void;
  variant?: "default" | "destructive";
}> = ({ active, icon, label, onClick, variant = "default" }) => (
  <button
    className={cn(
      "p-2 rounded-lg transition-all flex items-center justify-center",
      variant === "default" && [
        "text-white/70 hover:text-white hover:bg-white/10",
        active && "bg-primary/20 text-primary hover:bg-primary/30 hover:text-primary",
      ],
      variant === "destructive" && "bg-destructive/10 text-destructive hover:bg-destructive/20 hover:scale-105",
    )}
    onClick={onClick}
    title={label}
  >
    {icon}
  </button>
);

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
    <div className="absolute top-0 left-0 w-full z-[100] flex items-center justify-between px-8 py-6 bg-gradient-to-b from-black/80 to-transparent">
      <div className="flex items-center gap-8">
        {/* Info */}
        <div className="flex flex-col">
          <span className="text-lg font-bold tracking-tight text-white">{label}</span>
          <span className="text-[11px] font-medium uppercase tracking-widest text-white/40">
            Item {activeIndex + 1} of {totalItems}
          </span>
        </div>

        {/* Action Groups */}
        <div className="flex items-center gap-1.5 p-1 bg-white/5 backdrop-blur-md border border-white/10 rounded-xl">
          {isImage && (
            <>
              <ToolbarButton icon={<ZoomIn size={18} />} label="Zoom In" onClick={onZoomIn} />
              <ToolbarButton icon={<ZoomOut size={18} />} label="Zoom Out" onClick={onZoomOut} />
              <ToolbarButton icon={<RotateCw size={18} />} label="Rotate" onClick={onRotate} />
              <ToolbarButton icon={<Focus size={18} />} label="Reset View" onClick={onReset} />
            </>
          )}
          {isVideo && (
            <>
              <ToolbarButton
                active={videoMode === VideoMode.FIT}
                icon={<Minimize size={18} />}
                label="Fit to View"
                onClick={() => {
                  onSetVideoMode(VideoMode.FIT);
                }}
              />
              <ToolbarButton
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

      <ToolbarButton icon={<X size={20} />} label="Close (ESC)" onClick={onClose} variant="destructive" />
    </div>
  );
};
