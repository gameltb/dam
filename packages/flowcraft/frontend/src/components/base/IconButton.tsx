import React from "react";

interface IconButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  icon: React.ReactNode;
  label?: string;
  active?: boolean;
}

export const IconButton: React.FC<IconButtonProps> = ({
  icon,
  label,
  active,
  style,
  ...props
}) => {
  return (
    <button
      {...props}
      style={{
        background: active
          ? "rgba(100, 108, 255, 0.2)"
          : "rgba(255, 255, 255, 0.05)",
        border: "1px solid",
        borderColor: active
          ? "rgba(100, 108, 255, 0.4)"
          : "rgba(255, 255, 255, 0.1)",
        color: active ? "#646cff" : "white",
        width: "32px",
        height: "32px",
        borderRadius: "6px",
        cursor: "pointer",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontSize: "16px",
        transition: "all 0.2s cubic-bezier(0.4, 0, 0.2, 1)",
        outline: "none",
        ...style,
      }}
      className={`icon-button ${props.className ?? ""}`}
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
