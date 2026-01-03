import React from "react";

export interface MenuContainerProps {
  x: number;
  y: number;
  children: React.ReactNode;
}

export const MenuContainer: React.FC<MenuContainerProps> = ({
  x,
  y,
  children,
}) => (
  <div
    className="context-menu-container"
    style={{
      position: "fixed",
      top: y,
      left: x,
      backgroundColor: "var(--panel-bg)",
      border: "1px solid var(--node-border)",
      borderRadius: "8px",
      boxShadow: "0 10px 25px rgba(0,0,0,0.3)",
      zIndex: 1000,
      minWidth: "160px",
      padding: "4px 0",
      animation: "fade-in 0.1s ease-out",
      backdropFilter: "blur(10px)",
    }}
  >
    <style>{`
      @keyframes fade-in { from { opacity: 0; transform: scale(0.95); } to { opacity: 1; transform: scale(1); } }
    `}</style>
    {children}
  </div>
);
