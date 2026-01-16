import React from "react";

export interface CheckboxFieldProps {
  label: string;
  onChange: (value: boolean) => void;
  value: boolean;
}

export const CheckboxField: React.FC<CheckboxFieldProps> = ({ label, onChange, value }) => {
  return (
    <div className="widget checkbox-field" style={{ alignItems: "center", display: "flex", gap: "8px" }}>
      <input
        checked={value}
        className="nodrag"
        onChange={(e) => {
          onChange(e.target.checked);
        }}
        type="checkbox"
      />
      <label style={{ fontSize: "13px" }}>{label}</label>
    </div>
  );
};
