import React, { useState } from "react";

interface MarkdownRendererProps {
  content: string;
  onEdit?: (newContent: string) => void;
}

export const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({ content, onEdit }) => {
  const [isEditing, setIsEditing] = useState(false);
  const [localValue, setLocalValue] = useState(content);

  const handleDoubleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    setIsEditing(true);
  };

  const handleBlur = () => {
    setIsEditing(false);
    if (localValue !== content && onEdit) {
      onEdit(localValue);
    }
  };

  if (isEditing) {
    return (
      <textarea
        autoFocus
        className="nodrag nopan"
        onBlur={handleBlur}
        onChange={(e) => {
          setLocalValue(e.target.value);
        }}
        style={{
          backgroundColor: "#1e1e1e",
          border: "none",
          borderRadius: "inherit",
          boxSizing: "border-box",
          color: "#d4d4d4",
          fontFamily: "monospace",
          fontSize: "13px",
          height: "100%",
          outline: "none",
          padding: "10px",
          resize: "none",
          width: "100%",
        }}
        value={localValue}
      />
    );
  }

  return (
    <div
      onDoubleClick={handleDoubleClick}
      style={{
        backgroundColor: "#1a1a1a",
        borderRadius: "inherit",
        boxSizing: "border-box",
        color: "#e0e0e0",
        cursor: "text",
        fontSize: "14px",
        height: "100%",
        lineHeight: "1.6",
        overflowY: "auto",
        padding: "12px",
        width: "100%",
      }}
    >
      {/* Basic MD rendering simulation for now, can be replaced with a real MD parser later */}
      {content.split("\n").map((line, i) => {
        if (line.startsWith("# "))
          return (
            <h1 key={i} style={{ marginTop: 0 }}>
              {line.slice(2)}
            </h1>
          );
        if (line.startsWith("## ")) return <h2 key={i}>{line.slice(3)}</h2>;
        if (line.startsWith("- ")) return <li key={i}>{line.slice(2)}</li>;
        return (
          <p key={i} style={{ margin: "0 0 8px 0" }}>
            {line}
          </p>
        );
      })}
    </div>
  );
};
