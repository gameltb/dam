import React, { useMemo } from "react";
import { buildActionTree } from "../../utils/menuUtils";
import { GenericSubMenu } from "../base/SubMenu";
import { MenuContainer } from "./MenuContainer";

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
      <div className="fc-menu-section">
        {onFocus && (
          <div className="fc-menu-item" onClick={() => { onFocus(); onClose(); }}>
            🔍 Focus View
          </div>
        )}
        {onOpenEditor && (
          <div className="fc-menu-item" onClick={() => { onOpenEditor(); onClose(); }}>
            🎨 Open Editor
          </div>
        )}
        {onCopy && (
          <div className="fc-menu-item" onClick={() => { onCopy(); onClose(); }}>
            📋 Copy (Ctrl+C)
          </div>
        )}
        {onDuplicate && (
          <div className="fc-menu-item" onClick={() => { onDuplicate(); onClose(); }}>
            👯 Duplicate (Ctrl+D)
          </div>
        )}
        {onGroupSelected && (
          <div className="fc-menu-item" onClick={() => { onGroupSelected(); onClose(); }}>
            📦 Group Selected
          </div>
        )}
        {onLayoutGroup && (
          <div className="fc-menu-item" onClick={() => { onLayoutGroup(); onClose(); }}>
            📐 Layout Group
          </div>
        )}
        <div className="fc-menu-item text-red-400" onClick={() => { onDelete(); onClose(); }}>
          🗑️ Delete
        </div>
      </div>

      {actionTree.length > 0 && (
        <div className="fc-menu-section">
          <div className="fc-menu-label">Server Actions</div>
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
                  className="fc-menu-item"
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