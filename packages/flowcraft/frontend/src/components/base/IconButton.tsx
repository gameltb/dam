import React from "react";

interface IconButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  active?: boolean;
  icon: React.ReactNode;
  label?: string;
}

export const IconButton: React.FC<IconButtonProps> = ({ active, icon, label, style, ...props }) => {
  return (
    <button
      {...props}
      className={`icon-button ${props.className ?? ""}`}
      style={{
        alignItems: "center",
        background: active ? "rgba(100, 108, 255, 0.2)" : "rgba(255, 255, 255, 0.05)",
        border: "1px solid",
        borderColor: active ? "rgba(100, 108, 255, 0.4)" : "rgba(255, 255, 255, 0.1)",
        borderRadius: "6px",
        color: active ? "#646cff" : "white",
        cursor: "pointer",
        display: "flex",
        fontSize: "16px",
        height: "32px",
        justifyContent: "center",
        outline: "none",
        transition: "all 0.2s cubic-bezier(0.4, 0, 0.2, 1)",
        width: "32px",
        ...style,
      }}
      title={label}
    >
      {icon}
      <style>{`
        .icon-button:hover {
          background-color: rgba(255, 255, 255, 0.15) !important;
          border-color: rgba(255, 255, 255, 0.3) !important;
          transform: translateY(-1px);
        }
        .icon-button:active {
          transform: translateY(0);
        }
        .icon-button:disabled {
          opacity: 0.3;
          cursor: not-allowed;
          transform: none;
        }
      `}</style>
    </button>
  );
};
