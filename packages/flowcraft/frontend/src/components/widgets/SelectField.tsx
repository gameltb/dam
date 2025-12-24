import React from "react";

export interface SelectFieldProps {
  value: unknown;
  onChange: (value: unknown) => void;
  label?: string;
  options: { label: string; value: unknown }[];
}

export const SelectField: React.FC<SelectFieldProps> = ({
  value,
  onChange,
  label,
  options,
}) => {
  return (
    <div className="widget select-field">
      {label && (
        <label
          style={{ fontSize: "11px", display: "block", marginBottom: "2px" }}
        >
          {label}
        </label>
      )}
      <select
        className="nodrag"
        value={String(value)}
        onChange={(e) => onChange(e.target.value)}
        style={{
          width: "100%",
          padding: "4px",
          borderRadius: "4px",
          border: "1px solid #ccc",
        }}
      >
        {options.map((opt) => (
          <option key={String(opt.value)} value={String(opt.value)}>
            {opt.label}
          </option>
        ))}
      </select>
    </div>
  );
};
