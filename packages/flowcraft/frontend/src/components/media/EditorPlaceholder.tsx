import React from "react";
import { type AppNode, isDynamicNode } from "../../types";

interface EditorPlaceholderProps {
  node: AppNode;
  onClose: () => void;
}

export const EditorPlaceholder: React.FC<EditorPlaceholderProps> = ({
  node,
  onClose,
}) => {
  if (!isDynamicNode(node)) return null;
  const media = node.data.media;

  return (
    <div
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        width: "100vw",
        height: "100vh",
        backgroundColor: "rgba(15, 15, 20, 0.98)",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        zIndex: 4000,
        color: "white",
        fontFamily: "sans-serif",
      }}
    >
      <div
        style={{
          textAlign: "center",
          padding: "40px",
          border: "1px dashed rgba(255,255,255,0.2)",
          borderRadius: "20px",
          maxWidth: "500px",
        }}
      >
        <div style={{ fontSize: "48px", marginBottom: "20px" }}>🎨</div>
        <h2 style={{ margin: "0 0 10px 0" }}>
          {media?.type === "video" ? "Video" : "Image"} Editor
        </h2>
        <p style={{ opacity: 0.6, lineHeight: "1.5" }}>
          This is a placeholder for the advanced web-based {media?.type} editor.
          Soon you will be able to crop, filter, and modify your assets directly
          here.
        </p>
        <div style={{ marginTop: "30px", fontSize: "13px", color: "#646cff" }}>
          Target Node ID:{" "}
          <code
            style={{
              background: "rgba(255,255,255,0.1)",
              padding: "2px 6px",
              borderRadius: "4px",
            }}
          >
            {node.id}
          </code>
        </div>
        <button
          onClick={onClose}
          style={{
            marginTop: "40px",
            padding: "12px 30px",
            backgroundColor: "#646cff",
            border: "none",
            color: "white",
            borderRadius: "8px",
            cursor: "pointer",
            fontWeight: "bold",
          }}
        >
          Return to Canvas
        </button>
      </div>
    </div>
  );
};
