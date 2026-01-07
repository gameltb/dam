import React from "react";

import { MenuContainer } from "./MenuContainer";

interface EdgeContextMenuProps {
  edgeId: string;
  onClose: () => void;
  onDelete: () => void;
  x: number;
  y: number;
}

export const EdgeContextMenu: React.FC<EdgeContextMenuProps> = ({
  onClose,
  onDelete,
  x,
  y,
}) => (
  <MenuContainer x={x} y={y}>
    <div className="fc-menu-section">
      <div
        className="fc-menu-item text-red-400"
        onClick={() => {
          onDelete();
          onClose();
        }}
      >
        🗑️ Delete Edge
      </div>
    </div>
  </MenuContainer>
);
