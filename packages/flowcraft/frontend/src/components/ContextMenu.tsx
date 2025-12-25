import React, { memo } from "react";
import { type NodeTemplate } from "../types";

export interface ContextMenuProps {
  x: number;
  y: number;
  onClose: () => void;
  onDelete?: () => void;
  onDeleteEdge?: () => void;
  onFocus?: () => void;
  onOpenEditor?: () => void;
  onCopy?: () => void;
  onPaste?: () => void;
  onDuplicate?: () => void;
  dynamicActions?: { id: string; name: string; onClick: () => void }[];
  onToggleTheme: () => void;
  templates: NodeTemplate[];
  onAddNode: (template: NodeTemplate) => void;
  onAutoLayout: () => void;
  onGroupSelected?: () => void;
  onLayoutGroup?: () => void;
  onGalleryAction?: (url: string) => void;
  galleryItemUrl?: string;
  isPaneMenu?: boolean;
}

export const ContextMenu: React.FC<ContextMenuProps> = ({
  x,
  y,
  onClose,
  onDelete,
  onDeleteEdge,
  onFocus,
  onOpenEditor,
  onCopy,
  onPaste,
  onDuplicate,
  dynamicActions = [],
  onToggleTheme,
  templates,
  onAddNode,
  onAutoLayout,
  onGroupSelected,
  onLayoutGroup,
  onGalleryAction,
  galleryItemUrl,
  isPaneMenu,
}) => {
  const itemStyle: React.CSSProperties = {
    padding: "8px 12px",
    cursor: "pointer",
    fontSize: "12px",
    color: "var(--text-color)",
    display: "flex",
    alignItems: "center",
    gap: "8px",
    transition: "background 0.2s",
  };

  const sectionStyle: React.CSSProperties = {
    borderBottom: "1px solid var(--node-border)",
    paddingBottom: "4px",
    marginBottom: "4px",
  };

  const handleMouseEnter = (e: React.MouseEvent) => {
    (e.currentTarget as HTMLElement).style.backgroundColor =
      "rgba(100, 108, 255, 0.15)";
  };

  const handleMouseLeave = (e: React.MouseEvent) => {
    (e.currentTarget as HTMLElement).style.backgroundColor = "transparent";
  };

  return (
    <div
      style={{
        position: "fixed",
        top: y,
        left: x,
        backgroundColor: "var(--panel-bg)",
        border: "1px solid var(--node-border)",
        borderRadius: "8px",
        boxShadow: "0 10px 25px rgba(0,0,0,0.3)",
        zIndex: 1000,
        minWidth: "160px",
        padding: "4px 0",
        animation: "fade-in 0.1s ease-out",
        backdropFilter: "blur(10px)",
      }}
      onMouseLeave={onClose}
    >
      <style>{`
        @keyframes fade-in { from { opacity: 0; transform: scale(0.95); } to { opacity: 1; transform: scale(1); } }
      `}</style>

      {/* --- Node/Edge Specific Actions --- */}
      {(onDelete ||
        onDeleteEdge ||
        onFocus ||
        onOpenEditor ||
        onCopy ||
        onDuplicate) && (
        <div style={sectionStyle}>
          {onFocus && (
            <div
              style={itemStyle}
              onMouseEnter={handleMouseEnter}
              onMouseLeave={handleMouseLeave}
              onClick={onFocus}
            >
              🔍 Focus View
            </div>
          )}
          {onOpenEditor && (
            <div
              style={itemStyle}
              onMouseEnter={handleMouseEnter}
              onMouseLeave={handleMouseLeave}
              onClick={onOpenEditor}
            >
              🎨 Open Editor
            </div>
          )}
          {onCopy && (
            <div
              style={itemStyle}
              onMouseEnter={handleMouseEnter}
              onMouseLeave={handleMouseLeave}
              onClick={() => {
                onCopy();
                onClose();
              }}
            >
              📋 Copy (Ctrl+C)
            </div>
          )}
          {onDuplicate && (
            <div
              style={itemStyle}
              onMouseEnter={handleMouseEnter}
              onMouseLeave={handleMouseLeave}
              onClick={() => {
                onDuplicate();
                onClose();
              }}
            >
              👯 Duplicate (Ctrl+D)
            </div>
          )}
          {(onDelete || onDeleteEdge) && (
            <div
              style={{ ...itemStyle, color: "#f87171" }}
              onMouseEnter={handleMouseEnter}
              onMouseLeave={handleMouseLeave}
              onClick={onDelete || onDeleteEdge}
            >
              🗑️ Delete
            </div>
          )}
        </div>
      )}

      {/* --- Global/Pane Actions --- */}
      {isPaneMenu && (
        <div style={sectionStyle}>
          {onPaste && (
            <div
              style={itemStyle}
              onMouseEnter={handleMouseEnter}
              onMouseLeave={handleMouseLeave}
              onClick={() => {
                onPaste();
                onClose();
              }}
            >
              📥 Paste (Ctrl+V)
            </div>
          )}
          <div
            style={itemStyle}
            onMouseEnter={handleMouseEnter}
            onMouseLeave={handleMouseLeave}
            onClick={onAutoLayout}
          >
            🪄 Auto Layout
          </div>
          {onGroupSelected && (
            <div
              style={itemStyle}
              onMouseEnter={handleMouseEnter}
              onMouseLeave={handleMouseLeave}
              onClick={onGroupSelected}
            >
              📦 Group Selected
            </div>
          )}
        </div>
      )}

      {/* --- Subgraph Actions --- */}
      {onLayoutGroup && (
        <div style={sectionStyle}>
          <div
            style={itemStyle}
            onMouseEnter={handleMouseEnter}
            onMouseLeave={handleMouseLeave}
            onClick={onLayoutGroup}
          >
            📐 Layout Group
          </div>
        </div>
      )}

      {/* --- Add Node Submenu --- */}
      {isPaneMenu && (
        <div style={sectionStyle}>
          <div
            style={{
              ...itemStyle,
              cursor: "default",
              color: "var(--sub-text)",
            }}
          >
            ADD NODE
          </div>
          {templates.map((tpl) => (
            <div
              key={tpl.id}
              style={itemStyle}
              onMouseEnter={handleMouseEnter}
              onMouseLeave={handleMouseLeave}
              onClick={() => {
                onAddNode(tpl);
                onClose();
              }}
            >
              + {tpl.label}
            </div>
          ))}
        </div>
      )}

      {/* --- Gallery Item Actions --- */}
      {galleryItemUrl && onGalleryAction && (
        <div style={sectionStyle}>
          <div
            style={itemStyle}
            onMouseEnter={handleMouseEnter}
            onMouseLeave={handleMouseLeave}
            onClick={() => {
              onGalleryAction(galleryItemUrl);
              onClose();
            }}
          >
            ✨ Extract Item to Node
          </div>
        </div>
      )}

      {/* --- Server Actions --- */}
      {dynamicActions.length > 0 && (
        <div style={sectionStyle}>
          <div
            style={{
              ...itemStyle,
              cursor: "default",
              color: "var(--sub-text)",
            }}
          >
            SERVER ACTIONS
          </div>
          {dynamicActions.map((action) => (
            <div
              key={action.id}
              style={itemStyle}
              onMouseEnter={handleMouseEnter}
              onMouseLeave={handleMouseLeave}
              onClick={action.onClick}
            >
              ⚡ {action.name}
            </div>
          ))}
        </div>
      )}

      {/* --- Bottom Helpers --- */}
      <div
        style={itemStyle}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
        onClick={() => {
          onToggleTheme();
          onClose();
        }}
      >
        🌓 Switch Theme
      </div>
    </div>
  );
};

export default memo(ContextMenu);
