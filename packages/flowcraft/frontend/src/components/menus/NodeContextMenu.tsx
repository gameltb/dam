import React, { useMemo } from "react";
import { buildActionTree } from "../../utils/menuUtils";
import { GenericSubMenu } from "../base/SubMenu";
import { MenuContainer } from "./MenuContainer";
import {
  itemStyle,
  sectionStyle,
  labelStyle,
  handleMouseEnter,
  handleMouseLeave,
} from "./MenuShared";

interface NodeContextMenuProps {
  x: number;
  y: number;
  nodeId: string;
  onDelete: () => void;
  onFocus?: () => void;
  onOpenEditor?: () => void;
  onCopy?: () => void;
  onDuplicate?: () => void;
  onClose: () => void;
  dynamicActions?: {
    id: string;
    name: string;
    onClick: () => void;
    path?: string[];
  }[];
  onLayoutGroup?: () => void;
  onGroupSelected?: () => void;
}

export const NodeContextMenu: React.FC<NodeContextMenuProps> = ({
  x,
  y,
  onDelete,
  onFocus,
  onOpenEditor,
  onCopy,
  onDuplicate,
  onClose,
  dynamicActions = [],
  onLayoutGroup,
  onGroupSelected,
}) => {
  const actionTree = useMemo(
    () => buildActionTree(dynamicActions),
    [dynamicActions],
  );

  return (
    <MenuContainer x={x} y={y}>
      <div style={sectionStyle}>
        {onFocus && (
          <div
            style={itemStyle}
            onMouseEnter={handleMouseEnter}
            onMouseLeave={handleMouseLeave}
            onClick={() => {
              onFocus();
              onClose();
            }}
          >
            🔍 Focus View
          </div>
        )}
        {onOpenEditor && (
          <div
            style={itemStyle}
            onMouseEnter={handleMouseEnter}
            onMouseLeave={handleMouseLeave}
            onClick={() => {
              onOpenEditor();
              onClose();
            }}
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
        {onLayoutGroup && (
          <div
            style={itemStyle}
            onMouseEnter={handleMouseEnter}
            onMouseLeave={handleMouseLeave}
            onClick={() => {
              onLayoutGroup();
              onClose();
            }}
          >
            📐 Layout Group
          </div>
        )}
        <div
          style={{ ...itemStyle, color: "#f87171" }}
          onMouseEnter={handleMouseEnter}
          onMouseLeave={handleMouseLeave}
          onClick={() => {
            onDelete();
            onClose();
          }}
        >
          🗑️ Delete
        </div>
      </div>

      {actionTree.length > 0 && (
        <div style={sectionStyle}>
          <div style={labelStyle}>SERVER ACTIONS</div>
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
                  onClick={() => {
                    action.onClick();
                    onClose();
                  }}
                >
                  ⚡ {node.label}
                </div>
              );
            }
            return null;
          })}
        </div>
      )}
    </MenuContainer>
  );
};
