import React from 'react';

type ContextMenuProps = {
  x: number;
  y: number;
  onClose: () => void;
  onDelete: () => void;
  onFocus: () => void;
  onShowDamEntity: () => void;
};

export function ContextMenu({ x, y, onClose, onDelete, onFocus, onShowDamEntity }: ContextMenuProps) {
  return (
    <div
      style={{
        position: 'absolute',
        top: y,
        left: x,
        backgroundColor: 'white',
        border: '1px solid #ddd',
        borderRadius: 5,
        zIndex: 1000,
        boxShadow: '0 2px 5px rgba(0,0,0,0.15)',
      }}
      onClick={onClose}
    >
      <ul style={{ listStyle: 'none', margin: 0, padding: '5px 0' }}>
        <li
          onClick={onDelete}
          style={{ padding: '5px 15px', cursor: 'pointer' }}
          className="context-menu-item"
        >
          Delete
        </li>
        <li
          onClick={onFocus}
          style={{ padding: '5px 15px', cursor: 'pointer' }}
          className="context-menu-item"
        >
          Focus
        </li>
        <li
          onClick={onShowDamEntity}
          style={{ padding: '5px 15px', cursor: 'pointer' }}
          className="context-menu-item"
        >
          显示DAM实体
        </li>
      </ul>
    </div>
  );
}
