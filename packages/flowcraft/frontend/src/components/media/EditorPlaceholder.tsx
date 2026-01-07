import React from "react";

import { MediaType } from "../../generated/flowcraft/v1/core/node_pb";
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
        alignItems: "center",
        backgroundColor: "rgba(15, 15, 20, 0.98)",
        color: "white",
        display: "flex",
        fontFamily: "sans-serif",
        height: "100vh",
        justifyContent: "center",
        left: 0,
        position: "fixed",
        top: 0,
        width: "100vw",
        zIndex: 4000,
      }}
    >
      <div
        style={{
          border: "1px dashed rgba(255,255,255,0.2)",
          borderRadius: "20px",
          maxWidth: "500px",
          padding: "40px",
          textAlign: "center",
        }}
      >
        <div style={{ fontSize: "48px", marginBottom: "20px" }}>🎨</div>
        <h2 style={{ margin: "0 0 10px 0" }}>
          {media?.type === MediaType.MEDIA_VIDEO ? "Video" : "Image"} Editor
        </h2>
        <p style={{ lineHeight: "1.5", opacity: 0.6 }}>
          This is a placeholder for the advanced web-based {media?.type} editor.
          Soon you will be able to crop, filter, and modify your assets directly
          here.
        </p>
        <div style={{ color: "#646cff", fontSize: "13px", marginTop: "30px" }}>
          Target NODE_ID:{" "}
          <code
            style={{
              background: "rgba(255,255,255,0.1)",
              borderRadius: "4px",
              padding: "2px 6px",
            }}
          >
            {node.id}
          </code>
        </div>
        <button
          onClick={onClose}
          style={{
            backgroundColor: "#646cff",
            border: "none",
            borderRadius: "8px",
            color: "white",
            cursor: "pointer",
            fontWeight: "bold",
            marginTop: "40px",
            padding: "12px 30px",
          }}
        >
          Return to Canvas
        </button>
      </div>
    </div>
  );
};
