import React from "react";

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
  const handleLoadedMetadata = (e: React.SyntheticEvent<HTMLVideoElement>) => {
    const { videoHeight, videoWidth } = e.currentTarget;
    if (videoWidth && videoHeight && onDimensionsLoad) {
      onDimensionsLoad(videoWidth / videoHeight);
    }
  };

  return (
    <div
      style={{
        alignItems: "center",
        display: "flex",
        height: "100%",
        justifyContent: "center",
        overflow: "hidden",
        width: "100%",
      }}
    >
      <video
        autoPlay={autoPlay}
        controls={controls}
        draggable={false}
        loop
        muted={muted}
        onLoadedMetadata={handleLoadedMetadata}
        src={url}
        style={{ height: "100%", objectFit: "cover", width: "100%" }}
      />
    </div>
  );
};
