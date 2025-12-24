// src/components/media/ImageRenderer.tsx

import React from "react";

export interface ImageRendererProps {
  url: string;
}

export const ImageRenderer: React.FC<ImageRendererProps> = ({ url }) => {
  return (
    <img
      src={url}
      alt="media content"
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
