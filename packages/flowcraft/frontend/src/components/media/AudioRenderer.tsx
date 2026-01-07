import React from "react";

interface AudioRendererProps {
  autoPlay?: boolean;
  controls?: boolean;
  url: string;
}

export const AudioRenderer: React.FC<AudioRendererProps> = ({
  autoPlay = false,
  controls = true,
  url,
}) => {
  return (
    <div
      style={{
        alignItems: "center",
        backgroundColor: "#1a1a1a",
        borderRadius: "inherit",
        boxSizing: "border-box",
        containerType: "size",
        display: "flex",
        flexDirection: "column",
        gap: "8px",
        height: "100%",
        justifyContent: "center",
        overflow: "hidden",
        padding: "8px 12px",
        width: "100%",
      }}
    >
      {/* Audio Icon / Visualizer Placeholder - Hide if too short */}
      <div
        className="audio-icon"
        style={{
          alignItems: "center",
          backgroundColor: "rgba(255,255,255,0.1)",
          borderRadius: "50%",
          display: "flex",
          flex: 1,
          fontSize: "20px",
          justifyContent: "center",
          maxHeight: "48px",
          maxWidth: "48px",
          minHeight: "32px",
          minWidth: "32px",
          overflow: "hidden",
        }}
      >
        🎵
      </div>
      <audio
        autoPlay={autoPlay}
        controls={controls}
        src={url}
        style={{ height: "32px", minHeight: "32px", width: "100%" }}
      />
      <style>{`
        /* Hide icon if node height is too small to fit both safely */
        @container (max-height: 80px) {
          .audio-icon { display: none !important; }
        }
      `}</style>
    </div>
  );
};
