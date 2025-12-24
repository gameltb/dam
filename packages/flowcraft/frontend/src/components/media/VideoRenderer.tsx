import React from "react";

interface VideoRendererProps {
  url: string;
  autoPlay?: boolean;
  muted?: boolean;
  controls?: boolean;
}

export const VideoRenderer: React.FC<VideoRendererProps> = ({
  url,
  autoPlay = false,
  muted = true,
  controls = false,
}) => {
  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        backgroundColor: "#000",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      <video
        src={url}
        autoPlay={autoPlay}
        muted={muted}
        controls={controls}
        loop
        style={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain" }}
      />
    </div>
  );
};
