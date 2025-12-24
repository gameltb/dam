import React, { useState, useEffect } from "react";

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
  const [localValue, setLocalValue] = useState(value);

  useEffect(() => {
    setLocalValue(value);
  }, [value]);

  const handleBlur = () => {
    if (localValue !== value) {
      onChange(localValue);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      if (localValue !== value) {
        onChange(localValue);
      }
      (e.target as HTMLInputElement).blur();
    }
  };

  return (
    <div className="widget text-field">
      {label && <label>{label}</label>}
      <input
        type="text"
        className="nodrag"
        value={localValue}
        onChange={(e) => setLocalValue(e.target.value)}
        onBlur={handleBlur}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        style={{ width: "100%", boxSizing: "border-box" }}
      />
    </div>
  );
};
