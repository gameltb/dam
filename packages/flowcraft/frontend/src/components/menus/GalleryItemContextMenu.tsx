import React from "react";
import { MenuContainer } from "./MenuContainer";
import {
  itemStyle,
  sectionStyle,
  handleMouseEnter,
  handleMouseLeave,
} from "./MenuShared";

interface GalleryItemContextMenuProps {
  x: number;
  y: number;
  url: string;
  onExtract: (url: string) => void;
  onClose: () => void;
}

export const GalleryItemContextMenu: React.FC<GalleryItemContextMenuProps> = ({
  x,
  y,
  url,
  onExtract,
  onClose,
}) => (
  <MenuContainer x={x} y={y}>
    <div style={sectionStyle}>
      <div
        style={itemStyle}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
        onClick={() => {
          onExtract(url);
          onClose();
        }}
      >
        ✨ Extract Item to Node
      </div>
    </div>
  </MenuContainer>
);
