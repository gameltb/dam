import React from "react";

interface AudioRendererProps {
  url: string;
  autoPlay?: boolean;
  controls?: boolean;
}

export const AudioRenderer: React.FC<AudioRendererProps> = ({
  url,
  autoPlay = false,
  controls = true,
}) => {
  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        backgroundColor: "#1a1a1a",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        padding: "8px 12px",
        boxSizing: "border-box",
        borderRadius: "inherit",
        gap: "8px",
        overflow: "hidden",
        containerType: "size",
      }}
    >
      {/* Audio Icon / Visualizer Placeholder - Hide if too short */}
      <div
        style={{
          minHeight: "32px",
          minWidth: "32px",
          maxHeight: "48px",
          maxWidth: "48px",
          flex: 1,
          borderRadius: "50%",
          backgroundColor: "rgba(255,255,255,0.1)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontSize: "20px",
          overflow: "hidden",
        }}
        className="audio-icon"
      >
        🎵
      </div>
      <audio
        src={url}
        autoPlay={autoPlay}
        controls={controls}
        style={{ width: "100%", height: "32px", minHeight: "32px" }}
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
