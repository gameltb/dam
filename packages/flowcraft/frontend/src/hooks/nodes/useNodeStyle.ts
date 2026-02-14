import { useMemo } from "react";

export function useNodeStyle(selected: boolean | undefined) {
  return useMemo((): React.CSSProperties => {
    return {
      backgroundColor: "var(--node-bg)",
      border: "1px solid",
      borderColor: selected ? "var(--primary-color)" : "var(--node-border)",
      borderRadius: "var(--radius-lg)",
      boxShadow: selected
        ? "0 0 0 1px var(--primary-color), 0 0 15px rgba(100, 108, 255, 0.4), 0 10px 25px rgba(0,0,0,0.4)"
        : "0 4px 12px rgba(0,0,0,0.2)",
      boxSizing: "border-box",
      color: "var(--text-color)",
      display: "flex",
      flexDirection: "column",
      height: "100%",
      position: "relative",
      transition: "border-color 0.2s ease, box-shadow 0.2s ease",
      width: "100%",
    };
  }, [selected]);
}
