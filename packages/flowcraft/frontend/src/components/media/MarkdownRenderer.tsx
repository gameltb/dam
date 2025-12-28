import React, { useState } from "react";

interface MarkdownRendererProps {
  content: string;
  onEdit?: (newContent: string) => void;
}

export const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({
  content,
  onEdit,
}) => {
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
        className="nodrag nopan"
        autoFocus
        value={localValue}
        onChange={(e) => {
          setLocalValue(e.target.value);
        }}
        onBlur={handleBlur}
        style={{
          width: "100%",
          height: "100%",
          backgroundColor: "#1e1e1e",
          color: "#d4d4d4",
          border: "none",
          padding: "10px",
          fontFamily: "monospace",
          fontSize: "13px",
          resize: "none",
          outline: "none",
          boxSizing: "border-box",
          borderRadius: "inherit",
        }}
      />
    );
  }

  return (
    <div
      onDoubleClick={handleDoubleClick}
      style={{
        width: "100%",
        height: "100%",
        padding: "12px",
        overflowY: "auto",
        fontSize: "14px",
        lineHeight: "1.6",
        color: "#e0e0e0",
        backgroundColor: "#1a1a1a",
        boxSizing: "border-box",
        cursor: "text",
        borderRadius: "inherit",
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
