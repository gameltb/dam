import { Play } from "lucide-react";
import React, { useState } from "react";

import { cn } from "@/lib/utils";

interface VideoRendererProps {
  autoPlay?: boolean;
  controls?: boolean;
  muted?: boolean;
  onDimensionsLoad?: (ratio: number) => void;
  url: string;
}

export const VideoRenderer: React.FC<VideoRendererProps> = ({
  autoPlay = false,
  controls = false,
  muted = true,
  onDimensionsLoad,
  url,
}) => {
  const [isHovered, setIsHovered] = useState(false);

  const handleLoadedMetadata = (e: React.SyntheticEvent<HTMLVideoElement>) => {
    const { videoHeight, videoWidth } = e.currentTarget;
    if (videoWidth && videoHeight && onDimensionsLoad) {
      onDimensionsLoad(videoWidth / videoHeight);
    }
  };

  return (
    <div
      className="relative w-full h-full group overflow-hidden bg-black flex items-center justify-center"
      onMouseEnter={() => {
        setIsHovered(true);
      }}
      onMouseLeave={() => {
        setIsHovered(false);
      }}
    >
      <video
        autoPlay={autoPlay}
        className="w-full h-full object-cover opacity-90 group-hover:opacity-100 transition-opacity"
        controls={controls}
        draggable={false}
        loop
        muted={muted}
        onLoadedMetadata={handleLoadedMetadata}
        src={url}
      />

      {!autoPlay && !controls && (
        <div
          className={cn(
            "absolute inset-0 flex items-center justify-center transition-opacity duration-300",
            isHovered ? "bg-black/20 opacity-100" : "opacity-0",
          )}
        >
          <div className="w-12 h-12 rounded-full bg-primary/80 text-white flex items-center justify-center shadow-xl backdrop-blur-sm transform transition-transform group-hover:scale-110">
            <Play fill="currentColor" size={24} />
          </div>
        </div>
      )}
    </div>
  );
};
