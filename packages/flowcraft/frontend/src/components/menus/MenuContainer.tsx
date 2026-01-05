import React from "react";
import { cn } from "../../lib/utils";

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
    className={cn(
      "fc-panel fixed z-[1000] min-w-[160px] py-1 shadow-2xl backdrop-blur-md animate-menu-in",
      "context-menu-container"
    )}
    style={{
      top: y,
      left: x,
    }}
  >
    {children}
  </div>
);