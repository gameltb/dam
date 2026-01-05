import React from "react";
import { MenuContainer } from "./MenuContainer";

interface EdgeContextMenuProps {
  x: number;
  y: number;
  edgeId: string;
  onDelete: () => void;
  onClose: () => void;
}

export const EdgeContextMenu: React.FC<EdgeContextMenuProps> = ({
  x,
  y,
  onDelete,
  onClose,
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