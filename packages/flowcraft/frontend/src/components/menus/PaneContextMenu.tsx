import React, { useMemo } from "react";
import { type NodeTemplate } from "../../types";
import { buildNodeTree } from "../../utils/menuUtils";
import { NodeSubMenu } from "../base/SubMenu";
import { MenuContainer } from "./MenuContainer";

interface PaneContextMenuProps {
  x: number;
  y: number;
  templates: NodeTemplate[];
  onAddNode: (template: NodeTemplate) => void;
  onAutoLayout: () => void;
  onPaste?: () => void;
  onCopy?: () => void;
  onDuplicate?: () => void;
  onClose: () => void;
  onGroupSelected?: () => void;
  onDeleteSelected?: () => void;
}

export const PaneContextMenu: React.FC<PaneContextMenuProps> = ({
  x,
  y,
  templates,
  onAddNode,
  onAutoLayout,
  onPaste,
  onCopy,
  onDuplicate,
  onClose,
  onGroupSelected,
  onDeleteSelected,
}) => {
  const menuTree = useMemo(() => buildNodeTree(templates), [templates]);

  return (
    <MenuContainer x={x} y={y}>
      <div className="fc-menu-section">
        {onCopy && (
          <div
            className="fc-menu-item"
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
            className="fc-menu-item"
            onClick={() => {
              onDuplicate();
              onClose();
            }}
          >
            👯 Duplicate (Ctrl+D)
          </div>
        )}
        {onPaste && (
          <div
            className="fc-menu-item"
            onClick={() => {
              onPaste();
              onClose();
            }}
          >
            📥 Paste (Ctrl+V)
          </div>
        )}
        <div
          className="fc-menu-item"
          onClick={() => {
            onAutoLayout();
            onClose();
          }}
        >
          🪄 Auto Layout
        </div>
        {onGroupSelected && (
          <div
            className="fc-menu-item"
            onClick={() => {
              onGroupSelected();
              onClose();
            }}
          >
            📦 Group Selected
          </div>
        )}
        {onDeleteSelected && (
          <div
            className="fc-menu-item text-red-400"
            onClick={() => {
              onDeleteSelected();
              onClose();
            }}
          >
            🗑️ Delete Selected
          </div>
        )}
      </div>

      <div className="fc-menu-section">
        <div className="fc-menu-label">Add Node</div>
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
                key={tpl.templateId}
                className="fc-menu-item"
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
    </MenuContainer>
  );
};
