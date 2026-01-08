import { create } from "@bufbuild/protobuf";
import React, { memo, useState } from "react";

import { NodeDataSchema } from "@/generated/flowcraft/v1/core/node_pb";
import { useFlowSocket } from "@/hooks/useFlowSocket";

interface NodeLabelProps {
  id: string;
  label: string;
  onChange: (id: string, label: string) => void;
  selected?: boolean;
}

export const NodeLabel: React.FC<NodeLabelProps> = memo(
  ({ id, label, onChange: _onChange, selected }) => {
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
        onDoubleClick={() => {
          setIsEditing(true);
        }}
        style={{
          alignItems: "center",

          backgroundColor: "rgba(0,0,0,0.05)",

          borderBottom: "1px solid var(--node-border)",

          borderRadius: "8px 8px 0 0",

          cursor: "text",

          display: "flex",

          justifyContent: "space-between",

          padding: "8px 12px",
        }}
      >
        {isEditing ? (
          <input
            autoFocus
            className="nodrag"
            onBlur={handleBlur}
            onChange={(e) => {
              setLocalLabel(e.target.value);
            }}
            onKeyDown={handleKeyDown}
            style={{
              background: "rgba(255,255,255,0.1)",

              border: "1px solid var(--primary-color)",

              borderRadius: "4px",

              color: "white",

              fontSize: "12px",

              fontWeight: "bold",

              outline: "none",

              padding: "2px 4px",

              width: "100%",
            }}
            value={localLabel}
          />
        ) : (
          <div
            style={{
              color: selected ? "var(--primary-color)" : "white",

              fontSize: "12px",

              fontWeight: "bold",

              maxWidth: "100%",

              overflow: "hidden",

              textOverflow: "ellipsis",

              whiteSpace: "nowrap",
            }}
          >
            {label}
          </div>
        )}
      </div>
    );
  },
);
