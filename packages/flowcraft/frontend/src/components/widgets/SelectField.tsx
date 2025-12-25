import React, { useState } from "react";

export interface SelectFieldProps {
  value: unknown;
  onChange: (value: unknown) => void;
  label?: string;
  options: { label: string; value: unknown }[];
  onFetchOptions?: () => Promise<{ label: string; value: unknown }[]>;
}

export const SelectField: React.FC<SelectFieldProps> = ({
  value,
  onChange,
  label,
  options: initialOptions,
  onFetchOptions,
}) => {
  const [options, setOptions] = useState(initialOptions);
  const [loading, setLoading] = useState(false);

  const handleFocus = async () => {
    if (onFetchOptions) {
      setLoading(true);
      try {
        const newOptions = await onFetchOptions();
        setOptions(newOptions);
      } catch (err) {
        console.error("Failed to fetch options", err);
      } finally {
        setLoading(false);
      }
    }
  };

  return (
    <div className="widget select-field">
      {label && (
        <label
          style={{ fontSize: "11px", display: "block", marginBottom: "2px" }}
        >
          {label} {loading && "(Loading...)"}
        </label>
      )}
      <select
        className="nodrag"
        value={String(value)}
        onChange={(e) => onChange(e.target.value)}
        onFocus={handleFocus}
        disabled={loading}
        style={{
          width: "100%",
          padding: "4px",
          borderRadius: "4px",
          border: "1px solid #ccc",
          opacity: loading ? 0.7 : 1,
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
