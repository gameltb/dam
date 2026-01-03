import React, { useMemo } from "react";
import { type NodeTemplate } from "../../types";
import { buildNodeTree } from "../../utils/menuUtils";
import { NodeSubMenu } from "../base/SubMenu";
import { MenuContainer } from "./MenuContainer";
import {
  itemStyle,
  sectionStyle,
  labelStyle,
  handleMouseEnter,
  handleMouseLeave,
} from "./MenuShared";

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
      <div style={sectionStyle}>
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
          onClick={() => {
            onAutoLayout();
            onClose();
          }}
        >
          🪄 Auto Layout
        </div>
        {onGroupSelected && (
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
        )}
        {onDeleteSelected && (
          <div
            style={{ ...itemStyle, color: "#f87171" }}
            onMouseEnter={handleMouseEnter}
            onMouseLeave={handleMouseLeave}
            onClick={() => {
              onDeleteSelected();
              onClose();
            }}
          >
            🗑️ Delete Selected
          </div>
        )}
      </div>

      <div style={sectionStyle}>
        <div style={labelStyle}>ADD NODE</div>
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
    </MenuContainer>
  );
};
