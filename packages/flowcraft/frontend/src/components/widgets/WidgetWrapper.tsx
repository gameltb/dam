// src/components/widgets/WidgetWrapper.tsx

import React, { useState } from "react";

export interface WidgetWrapperProps {
  children: React.ReactNode;
  isSwitchable: boolean;
  onToggleMode: () => void;
  onClick?: () => void;
}

export const WidgetWrapper: React.FC<WidgetWrapperProps> = ({
  children,
  isSwitchable,
  onToggleMode,
  onClick,
}) => {
  const [isSelected, setIsSelected] = useState(false);
  const [contextMenu, setContextMenu] = useState<{
    x: number;
    y: number;
  } | null>(null);

  const handleSelect = () => {
    setIsSelected(true);
  };

  const handleDeselect = () => {
    setIsSelected(false);
  };

  const handleContextMenu = (event: React.MouseEvent) => {
    event.preventDefault();
    if (isSwitchable) {
      setContextMenu({ x: event.clientX, y: event.clientY });
    }
  };

  const closeContextMenu = () => {
    setContextMenu(null);
  };

  const wrapperStyle: React.CSSProperties = {
    position: "relative",
    padding: "5px",
    border: isSelected ? "1px solid #777" : "1px solid transparent",
    borderRadius: "4px",
  };

  const buttonStyle: React.CSSProperties = {
    position: "absolute",
    top: "-10px",
    right: "-10px",
    cursor: "pointer",
    background: "#eee",
    border: "1px solid #ccc",
    borderRadius: "50%",
    width: "20px",
    height: "20px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    zIndex: 10,
  };

  return (
    <div
      style={wrapperStyle}
      onClick={() => {
        handleSelect();
        onClick?.();
      }}
      onBlur={handleDeselect}
      onContextMenu={handleContextMenu}
      tabIndex={0} // Make it focusable
    >
      {children}
      {isSelected && isSwitchable && (
        <button style={buttonStyle} onClick={onToggleMode} title="Switch mode">
          ⇆
        </button>
      )}
      {contextMenu && (
        <div
          style={{
            position: "fixed",
            top: contextMenu.y,
            left: contextMenu.x,
            background: "white",
            border: "1px solid #ccc",
            zIndex: 1000,
            padding: "5px",
          }}
          onMouseLeave={closeContextMenu}
        >
          <div
            onClick={() => {
              onToggleMode();
              closeContextMenu();
            }}
            style={{ cursor: "pointer", padding: "5px 10px" }}
          >
            Switch Mode
          </div>
        </div>
      )}
    </div>
  );
};
