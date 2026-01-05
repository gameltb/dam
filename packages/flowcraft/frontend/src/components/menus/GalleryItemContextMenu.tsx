import React from "react";
import { MenuContainer } from "./MenuContainer";

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
    <div className="fc-menu-section">
      <div
        className="fc-menu-item"
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