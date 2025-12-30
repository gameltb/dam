import React from "react";

interface VideoRendererProps {
  url: string;
  autoPlay?: boolean;
  muted?: boolean;
  controls?: boolean;
  onDimensionsLoad?: (ratio: number) => void;
}

export const VideoRenderer: React.FC<VideoRendererProps> = ({
  url,
  autoPlay = false,
  muted = true,
  controls = false,
  onDimensionsLoad,
}) => {
  const handleLoadedMetadata = (e: React.SyntheticEvent<HTMLVideoElement>) => {
    const { videoWidth, videoHeight } = e.currentTarget;
    if (videoWidth && videoHeight && onDimensionsLoad) {
      onDimensionsLoad(videoWidth / videoHeight);
    }
  };

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
        onLoadedMetadata={handleLoadedMetadata}
        loop
        draggable={false}
        style={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain" }}
      />
    </div>
  );
};
