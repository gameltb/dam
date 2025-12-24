import React from "react";

export interface CheckboxFieldProps {
  value: boolean;
  onChange: (value: boolean) => void;
  label: string;
}

export const CheckboxField: React.FC<CheckboxFieldProps> = ({
  value,
  onChange,
  label,
}) => {
  return (
    <div
      className="widget checkbox-field"
      style={{ display: "flex", alignItems: "center", gap: "8px" }}
    >
      <input
        type="checkbox"
        className="nodrag"
        checked={value}
        onChange={(e) => onChange(e.target.checked)}
      />
      <label style={{ fontSize: "13px" }}>{label}</label>
    </div>
  );
};
