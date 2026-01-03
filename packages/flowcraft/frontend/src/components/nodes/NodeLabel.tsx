import React, { useState, memo } from "react";
import { useFlowSocket } from "../../hooks/useFlowSocket";
import { create } from "@bufbuild/protobuf";
import { NodeDataSchema } from "../../generated/flowcraft/v1/core/node_pb";

interface NodeLabelProps {
  id: string;
  label: string;
  selected?: boolean;
  onChange: (id: string, label: string) => void;
}

export const NodeLabel: React.FC<NodeLabelProps> = memo(
  ({ id, label, selected, onChange: _onChange }) => {
    const { updateNodeData } = useFlowSocket({ disablePolling: true });

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

      if (!selected && isEditing) {
        setIsEditing(false);

        if (localLabel !== label) {
          updateNodeData(
            id,
            create(NodeDataSchema, { displayName: localLabel }),
          );
        }
      }
    }

    const handleBlur = () => {
      setIsEditing(false);

      if (localLabel !== label) {
        updateNodeData(id, create(NodeDataSchema, { displayName: localLabel }));
      }
    };

    const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (e.key === "Enter") {
        e.currentTarget.blur();
      } else if (e.key === "Escape") {
        setLocalLabel(label);

        setIsEditing(false);
      }
    };

    return (
      <div
        style={{
          padding: "8px 12px",

          borderBottom: "1px solid var(--node-border)",

          display: "flex",

          alignItems: "center",

          justifyContent: "space-between",

          backgroundColor: "rgba(0,0,0,0.05)",

          borderRadius: "8px 8px 0 0",

          cursor: "text",
        }}
        onDoubleClick={() => {
          setIsEditing(true);
        }}
      >
        {isEditing ? (
          <input
            autoFocus
            value={localLabel}
            onChange={(e) => {
              setLocalLabel(e.target.value);
            }}
            onBlur={handleBlur}
            onKeyDown={handleKeyDown}
            className="nodrag"
            style={{
              background: "rgba(255,255,255,0.1)",

              border: "1px solid var(--primary-color)",

              color: "white",

              fontSize: "12px",

              fontWeight: "bold",

              width: "100%",

              padding: "2px 4px",

              borderRadius: "4px",

              outline: "none",
            }}
          />
        ) : (
          <div
            style={{
              fontSize: "12px",

              fontWeight: "bold",

              color: selected ? "var(--primary-color)" : "white",

              overflow: "hidden",

              textOverflow: "ellipsis",

              whiteSpace: "nowrap",

              maxWidth: "100%",
            }}
          >
            {label}
          </div>
        )}
      </div>
    );
  },
);
