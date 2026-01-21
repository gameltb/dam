import React, { memo, useEffect, useState } from "react";

export interface TextFieldProps {
  label?: string;
  onChange: (value: string) => void;
  placeholder?: string;
  value: string;
}

export const TextField: React.FC<TextFieldProps> = memo(({ label, onChange, placeholder, value }) => {
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
        className="nodrag"
        onBlur={handleBlur}
        onChange={(e) => {
          setLocalValue(e.target.value);
        }}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        style={{ boxSizing: "border-box", width: "100%" }}
        type="text"
        value={localValue}
      />
    </div>
  );
});
