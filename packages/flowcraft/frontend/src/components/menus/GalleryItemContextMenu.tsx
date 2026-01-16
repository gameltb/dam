import React from "react";

import { MenuContainer } from "./MenuContainer";

interface GalleryItemContextMenuProps {
  onClose: () => void;
  onExtract: (url: string) => void;
  url: string;
  x: number;
  y: number;
}

export const GalleryItemContextMenu: React.FC<GalleryItemContextMenuProps> = ({ onClose, onExtract, url, x, y }) => (
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
