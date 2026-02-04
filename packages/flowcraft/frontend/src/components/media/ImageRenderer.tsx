import React, { useState } from "react";

import { cn } from "@/lib/utils";

export interface ImageRendererProps {
  onDimensionsLoad?: (ratio: number) => void;
  url: string;
}

export const ImageRenderer: React.FC<ImageRendererProps> = ({ onDimensionsLoad, url }) => {
  const [isLoaded, setIsLoaded] = useState(false);

  const handleLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
    const { naturalHeight, naturalWidth } = e.currentTarget;
    if (naturalWidth && naturalHeight && onDimensionsLoad) {
      onDimensionsLoad(naturalWidth / naturalHeight);
    }
    setIsLoaded(true);
  };

  return (
    <div className="relative w-full h-full overflow-hidden flex items-center justify-center bg-muted/10">
      {!isLoaded && (
        <div className="absolute inset-0 animate-pulse bg-muted/20 flex items-center justify-center">
          <div className="w-8 h-8 rounded-full border-2 border-primary/20 border-t-primary animate-spin" />
        </div>
      )}
      <img
        alt="media content"
        className={cn(
          "max-w-full max-h-full object-contain transition-opacity duration-500",
          isLoaded ? "opacity-100" : "opacity-0",
        )}
        draggable={false}
        onLoad={handleLoad}
        src={url}
      />
    </div>
  );
};
