import React from "react";

import { cn } from "@/lib/utils";

export interface MenuContainerProps {
  children: React.ReactNode;
  x: number;
  y: number;
}

export const MenuContainer: React.FC<MenuContainerProps> = ({
  children,
  x,
  y,
}) => (
  <div
    className={cn(
      "fc-panel fixed z-[1000] min-w-[160px] py-1 shadow-2xl backdrop-blur-md animate-menu-in",
      "context-menu-container",
    )}
    style={{
      left: x,
      top: y,
    }}
  >
    {children}
  </div>
);
