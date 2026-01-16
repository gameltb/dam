import { useStore } from "@xyflow/react";
import React, { useState } from "react";

export interface WidgetWrapperProps {
  children: React.ReactNode;
  inputPortId?: string; // 如果设置了，将监听连线状态
  isSwitchable: boolean;
  nodeId?: string;
  onClick?: () => void;
  onToggleMode: () => void;
}

export const WidgetWrapper: React.FC<WidgetWrapperProps> = ({
  children,
  inputPortId,
  isSwitchable,
  nodeId,
  onClick,
  onToggleMode,
}) => {
  const [isSelected, setIsSelected] = useState(false);
  const [contextMenu, setContextMenu] = useState<null | {
    x: number;
    y: number;
  }>(null);

  // 监听是否有连线连接到该隐式端口
  const isConnected = useStore((s) => {
    if (!inputPortId || !nodeId) return false;
    return s.edges.some((e) => e.target === nodeId && e.targetHandle === inputPortId);
  });

  const handleSelect = () => {
    setIsSelected(true);
  };

  const handleDeselect = () => {
    setIsSelected(false);
  };

  const handleContextMenu = (event: React.MouseEvent) => {
    const target = event.target as HTMLElement;
    if (target.tagName === "INPUT" || target.tagName === "TEXTAREA") return;

    if (isSwitchable) {
      event.preventDefault();
      setContextMenu({ x: event.clientX, y: event.clientY });
    }
  };

  const closeContextMenu = () => {
    setContextMenu(null);
  };

  const wrapperStyle: React.CSSProperties = {
    border: isSelected ? "1px solid #777" : "1px solid transparent",
    borderRadius: "4px",
    padding: "5px",
    position: "relative",
    // Remove global opacity when connected to keep port clear
    transition: "background 0.2s",
  };

  const buttonStyle: React.CSSProperties = {
    alignItems: "center",
    background: "#eee",
    border: "1px solid #ccc",
    borderRadius: "50%",
    cursor: "pointer",
    display: "flex",
    height: "20px",
    justifyContent: "center",
    position: "absolute",
    right: "-10px",
    top: "-10px",
    width: "20px",
    zIndex: 10,
  };

  return (
    <div
      onBlur={handleDeselect}
      onClick={() => {
        handleSelect();
        onClick?.();
      }}
      onContextMenu={handleContextMenu}
      style={wrapperStyle}
      tabIndex={0}
    >
      {isConnected && (
        <div
          style={{
            color: "#646cff",
            fontSize: "10px",
            left: -15,
            position: "absolute",
            top: "50%",
            transform: "translateY(-50%)",
          }}
        >
          ➔
        </div>
      )}
      {children}
      {isSelected && isSwitchable && (
        <button onClick={onToggleMode} style={buttonStyle} title="Switch mode">
          ⇆
        </button>
      )}
      {contextMenu && (
        <div
          onMouseLeave={closeContextMenu}
          style={{
            background: "white",
            border: "1px solid #ccc",
            left: contextMenu.x,
            padding: "5px",
            position: "fixed",
            top: contextMenu.y,
            zIndex: 1000,
          }}
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
