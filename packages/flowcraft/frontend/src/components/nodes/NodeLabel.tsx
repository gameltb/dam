import React, { useState, memo } from "react";
import { useMockSocket } from "../../hooks/useMockSocket";

interface NodeLabelProps {
  id: string;
  label: string;
  selected?: boolean;
  onChange: (id: string, label: string) => void;
}

export const NodeLabel: React.FC<NodeLabelProps> = memo(
  ({ id, label, selected, onChange }) => {
    const { sendNodeUpdate } = useMockSocket({ disablePolling: true });

    const [isEditing, setIsEditing] = useState(false);

    const [localLabel, setLocalLabel] = useState(label);

    const [prevLabel, setPrevLabel] = useState(label);

    const [prevSelected, setPrevSelected] = useState(selected);

    // Adjust state when props change (React recommendation instead of useEffect for sync)

    if (label !== prevLabel && !isEditing) {
      setPrevLabel(label);

      setLocalLabel(label);
    }

    if (selected !== prevSelected) {
      setPrevSelected(selected);

      if (!selected) {
        setIsEditing(false);
      }
    }

    const handleExitEdit = () => {
      setIsEditing(false);
      if (localLabel !== label) {
        sendNodeUpdate(id, { label: localLabel }).catch((err: unknown) => {
          console.error("Failed to update node label", err);
        });
      }
    };
    return (
      <div
        onClick={(e) => {
          if (selected) {
            e.stopPropagation();
            setIsEditing(true);
          }
        }}
        onContextMenu={(e) => {
          if (isEditing) e.stopPropagation();
        }}
        onMouseDown={(e) => {
          if (isEditing) e.stopPropagation();
        }}
        style={{
          marginBottom: "8px",
          borderBottom: "1px solid var(--node-border)",
          padding: "10px 12px",
          display: "flex",
          alignItems: "center",
          minHeight: "38px",
          boxSizing: "border-box",
        }}
      >
        {isEditing ? (
          <input
            className="nodrag nopan"
            autoFocus
            style={{
              background: "var(--input-bg)",
              border: "none",
              color: "var(--text-color)",
              fontSize: "13px",
              fontWeight: "bold",
              width: "100%",
              outline: "none",
              padding: "2px 4px",
              borderRadius: "2px",
            }}
            value={localLabel}
            onChange={(e) => {
              setLocalLabel(e.target.value);
              onChange(id, e.target.value);
            }}
            onBlur={handleExitEdit}
            onKeyDown={(e) => {
              e.stopPropagation();
              if (e.key === "Enter") handleExitEdit();
              if (e.key === "Escape") {
                setLocalLabel(label);
                setIsEditing(false);
              }
            }}
          />
        ) : (
          <div
            style={{
              fontSize: "13px",
              fontWeight: "bold",
              color: "var(--text-color)",
              userSelect: "text",
              width: "100%",
            }}
          >
            {label}
          </div>
        )}
      </div>
    );
  },
);
