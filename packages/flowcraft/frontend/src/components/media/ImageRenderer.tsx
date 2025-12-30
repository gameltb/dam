import React from "react";

export interface ImageRendererProps {
  url: string;
  onDimensionsLoad?: (ratio: number) => void;
}

export const ImageRenderer: React.FC<ImageRendererProps> = ({
  url,
  onDimensionsLoad,
}) => {
  const handleLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
    const { naturalWidth, naturalHeight } = e.currentTarget;
    if (naturalWidth && naturalHeight && onDimensionsLoad) {
      onDimensionsLoad(naturalWidth / naturalHeight);
    }
  };

  return (
    <img
      src={url}
      alt="media content"
      onLoad={handleLoad}
      draggable={false}
      style={{
        width: "100%",
        height: "100%",
        objectFit: "cover",
        display: "block",
        borderRadius: "inherit",
      }}
    />
  );
};
