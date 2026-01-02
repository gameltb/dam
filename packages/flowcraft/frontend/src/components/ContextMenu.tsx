import React, { memo, useMemo } from "react";
import { type NodeTemplate } from "../types";
import { GenericSubMenu, NodeSubMenu } from "./base/SubMenu";
import { buildNodeTree, buildActionTree } from "../utils/menuUtils";

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
  templates: NodeTemplate[];
  onAddNode: (template: NodeTemplate) => void;
  onAutoLayout: () => void;
  onGroupSelected?: () => void;
  onLayoutGroup?: () => void;
  onGalleryAction?: (url: string) => void;
  galleryItemUrl?: string;
  isPaneMenu?: boolean;
}

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
  templates,
  onAddNode,
  onAutoLayout,
  onGroupSelected,
  onLayoutGroup,
  onGalleryAction,
  galleryItemUrl,
  isPaneMenu,
}) => {
  // Build node template tree
  const menuTree = useMemo(() => buildNodeTree(templates), [templates]);

  // Build server action tree
  const actionTree = useMemo(
    () => buildActionTree(dynamicActions),
    [dynamicActions],
  );

  return (
    <div
      className="context-menu-container"
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
    >
      <style>{`
        @keyframes fade-in { from { opacity: 0; transform: scale(0.95); } to { opacity: 1; transform: scale(1); } }
      `}</style>

      {/* --- Node/Edge Specific Actions --- */}
      {(onDelete ??
        onDeleteEdge ??
        onFocus ??
        onOpenEditor ??
        onCopy ??
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
          {(onDelete ?? onDeleteEdge) && (
            <div
              style={{ ...itemStyle, color: "#f87171" }}
              onMouseEnter={handleMouseEnter}
              onMouseLeave={handleMouseLeave}
              onClick={onDelete ?? onDeleteEdge}
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
        </div>
      )}

      {onGroupSelected && (
        <div style={sectionStyle}>
          <div
            style={itemStyle}
            onMouseEnter={handleMouseEnter}
            onMouseLeave={handleMouseLeave}
            onClick={() => {
              onGroupSelected();
              onClose();
            }}
          >
            📦 Group Selected
          </div>
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
          {menuTree.map((node, i) => {
            if (node.children && node.children.length > 0) {
              return (
                <NodeSubMenu
                  key={`${node.label}-${String(i)}`}
                  label={node.label}
                  nodes={node.children}
                  onAdd={(tpl) => {
                    onAddNode(tpl);
                    onClose();
                  }}
                />
              );
            }
            if (node.template) {
              const tpl = node.template;
              return (
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
                  + {node.label}
                </div>
              );
            }
            return null;
          })}
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
      {actionTree.length > 0 && (
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
          {actionTree.map((node, i) => {
            if (node.children && node.children.length > 0) {
              return (
                <GenericSubMenu
                  key={`action-${node.label}-${String(i)}`}
                  label={node.label}
                  nodes={node.children}
                />
              );
            }
            if (node.action) {
              const action = node.action;
              return (
                <div
                  key={action.id}
                  style={itemStyle}
                  onMouseEnter={handleMouseEnter}
                  onMouseLeave={handleMouseLeave}
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

export default memo(ContextMenu);