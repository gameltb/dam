// src/components/widgets/TextField.tsx

import React from "react";

export interface TextFieldProps {
  value: string;
  onChange: (value: string) => void;
  label?: string;
  placeholder?: string;
}

export const TextField: React.FC<TextFieldProps> = ({
  value,
  onChange,
  label,
  placeholder,
}) => {
  return (
    <div className="widget text-field">
      {label && <label>{label}</label>}
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        style={{ width: "100%", boxSizing: "border-box" }}
      />
    </div>
  );
};
