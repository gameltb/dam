import React from "react";

export interface ImageRendererProps {
  onDimensionsLoad?: (ratio: number) => void;
  url: string;
}

export const ImageRenderer: React.FC<ImageRendererProps> = ({ onDimensionsLoad, url }) => {
  const handleLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
    const { naturalHeight, naturalWidth } = e.currentTarget;
    if (naturalWidth && naturalHeight && onDimensionsLoad) {
      onDimensionsLoad(naturalWidth / naturalHeight);
    }
  };

  return (
    <img
      alt="media content"
      draggable={false}
      onLoad={handleLoad}
      src={url}
      style={{
        borderRadius: "inherit",
        display: "block",
        height: "100%",
        objectFit: "contain",
        width: "100%",
      }}
    />
  );
};
