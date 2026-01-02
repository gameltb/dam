import React, { useState } from "react";
import { type NodeTemplate } from "../../types";

export interface MenuNode {
  label: string;
  template?: NodeTemplate;
  action?: { id: string; name: string; onClick: () => void };
  children?: MenuNode[];
}

export const GenericSubMenu: React.FC<{
  label: string;
  nodes: MenuNode[];
  onAdd?: (tpl: NodeTemplate) => void;
  depth?: number;
}> = ({ label, nodes, onAdd, depth = 0 }) => {
  const [isOpen, setIsExpanded] = useState(false);

  const itemStyle: React.CSSProperties = {
    padding: "8px 12px",
    cursor: "pointer",
    fontSize: "12px",
    color: "var(--text-color)",
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    gap: "8px",
    transition: "background 0.2s",
    position: "relative",
    width: "100%",
    boxSizing: "border-box",
  };

  return (
    <div
      style={{ position: "relative", width: "100%" }}
      onMouseEnter={() => {
        setIsExpanded(true);
      }}
      onMouseLeave={() => {
        setIsExpanded(false);
      }}
    >
      <div
        style={{
          ...itemStyle,
          backgroundColor: isOpen ? "rgba(100, 108, 255, 0.15)" : "transparent",
        }}
      >
        <span>{label}</span>
        <span style={{ fontSize: "10px", opacity: 0.5 }}>▶</span>
      </div>

      {isOpen && (
        <div
          style={{
            position: "absolute",
            left: "100%",
            top: 0,
            backgroundColor: "var(--panel-bg)",
            border: "1px solid var(--node-border)",
            borderRadius: "8px",
            boxShadow: "0 10px 25px rgba(0,0,0,0.3)",
            minWidth: "160px",
            padding: "4px 0",
            backdropFilter: "blur(10px)",
            zIndex: 1001 + depth,
          }}
        >
          {nodes.map((node, i) => {
            if (node.children && node.children.length > 0) {
              return (
                <GenericSubMenu
                  key={`${node.label}-${String(i)}`}
                  label={node.label}
                  nodes={node.children}
                  onAdd={onAdd}
                  depth={depth + 1}
                />
              );
            }
            if (node.template && onAdd) {
              const tpl = node.template;
              return (
                <div
                  key={tpl.id}
                  style={itemStyle}
                  onMouseEnter={(e) =>
                    (e.currentTarget.style.backgroundColor =
                      "rgba(100, 108, 255, 0.15)")
                  }
                  onMouseLeave={(e) =>
                    (e.currentTarget.style.backgroundColor = "transparent")
                  }
                  onClick={() => {
                    onAdd(tpl);
                  }}
                >
                  + {node.label}
                </div>
              );
            }
            if (node.action) {
              const action = node.action;
              return (
                <div
                  key={action.id}
                  style={itemStyle}
                  onMouseEnter={(e) =>
                    (e.currentTarget.style.backgroundColor =
                      "rgba(100, 108, 255, 0.15)")
                  }
                  onMouseLeave={(e) =>
                    (e.currentTarget.style.backgroundColor = "transparent")
                  }
                  onClick={action.onClick}
                >
                  ⚡ {node.label}
                </div>
              );
            }
            return null;
          })}
        </div>
      )}
    </div>
  );
};

export const NodeSubMenu: React.FC<{
  label: string;
  nodes: MenuNode[];
  onAdd: (tpl: NodeTemplate) => void;
  depth?: number;
}> = ({ label, nodes, onAdd, depth = 0 }) => (
  <GenericSubMenu label={label} nodes={nodes} onAdd={onAdd} depth={depth} />
);
