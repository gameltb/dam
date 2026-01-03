import React from "react";
import { MenuContainer } from "./MenuContainer";
import {
  itemStyle,
  sectionStyle,
  handleMouseEnter,
  handleMouseLeave,
} from "./MenuShared";

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
    <div style={sectionStyle}>
      <div
        style={{ ...itemStyle, color: "#f87171" }}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
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
