import React, { useState } from "react";

import { type NodeTemplate } from "@/types";

export interface MenuNode {
  action?: { id: string; name: string; onClick: () => void };
  children?: MenuNode[];
  label: string;
  template?: NodeTemplate;
}

export const GenericSubMenu: React.FC<{
  depth?: number;
  label: string;
  nodes: MenuNode[];
  onAdd?: (tpl: NodeTemplate) => void;
}> = ({ depth = 0, label, nodes, onAdd }) => {
  const [isOpen, setIsExpanded] = useState(false);

  const itemStyle: React.CSSProperties = {
    alignItems: "center",
    boxSizing: "border-box",
    color: "var(--text-color)",
    cursor: "pointer",
    display: "flex",
    fontSize: "12px",
    gap: "8px",
    justifyContent: "space-between",
    padding: "8px 12px",
    position: "relative",
    transition: "background 0.2s",
    width: "100%",
  };

  return (
    <div
      onMouseEnter={() => {
        setIsExpanded(true);
      }}
      onMouseLeave={() => {
        setIsExpanded(false);
      }}
      style={{ position: "relative", width: "100%" }}
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
            backdropFilter: "blur(10px)",
            backgroundColor: "var(--panel-bg)",
            border: "1px solid var(--node-border)",
            borderRadius: "8px",
            boxShadow: "0 10px 25px rgba(0,0,0,0.3)",
            left: "100%",
            minWidth: "160px",
            padding: "4px 0",
            position: "absolute",
            top: 0,
            zIndex: 1001 + depth,
          }}
        >
          {nodes.map((node, i) => {
            if (node.children && node.children.length > 0) {
              return (
                <GenericSubMenu
                  depth={depth + 1}
                  key={`${node.label}-${String(i)}`}
                  label={node.label}
                  nodes={node.children}
                  onAdd={onAdd}
                />
              );
            }
            if (node.template && onAdd) {
              const tpl = node.template;
              return (
                <div
                  key={tpl.templateId}
                  onClick={() => {
                    onAdd(tpl);
                  }}
                  onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = "rgba(100, 108, 255, 0.15)")}
                  onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "transparent")}
                  style={itemStyle}
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
                  onClick={action.onClick}
                  onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = "rgba(100, 108, 255, 0.15)")}
                  onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "transparent")}
                  style={itemStyle}
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
  depth?: number;
  label: string;
  nodes: MenuNode[];
  onAdd: (tpl: NodeTemplate) => void;
}> = ({ depth = 0, label, nodes, onAdd }) => <GenericSubMenu depth={depth} label={label} nodes={nodes} onAdd={onAdd} />;
