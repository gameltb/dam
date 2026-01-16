import React, { useState } from "react";

export interface SelectFieldProps {
  label?: string;
  onChange: (value: unknown) => void;
  onFetchOptions?: () => Promise<{ label: string; value: unknown }[]>;
  options: { label: string; value: unknown }[];
  value: unknown;
}

export const SelectField: React.FC<SelectFieldProps> = ({
  label,
  onChange,
  onFetchOptions,
  options: initialOptions,
  value,
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
        <label style={{ display: "block", fontSize: "11px", marginBottom: "2px" }}>
          {label} {loading && "(Loading...)"}
        </label>
      )}
      <select
        className="nodrag"
        disabled={loading}
        onChange={(e) => {
          onChange(e.target.value);
        }}
        onFocus={() => {
          void handleFocus();
        }}
        style={{
          border: "1px solid #ccc",
          borderRadius: "4px",
          opacity: loading ? 0.7 : 1,
          padding: "4px",
          width: "100%",
        }}
        value={String(value)}
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
