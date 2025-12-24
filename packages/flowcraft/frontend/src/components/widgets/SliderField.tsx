import React, { useState, useEffect } from "react";

export interface SliderFieldProps {
  value: number;
  onChange: (value: number) => void;
  label: string;
  min?: number;
  max?: number;
}

export const SliderField: React.FC<SliderFieldProps> = ({
  value,
  onChange,
  label,
  min = 0,
  max = 100,
}) => {
  const [localValue, setLocalValue] = useState(value);

  // Sync with prop if it changes from outside
  useEffect(() => {
    setLocalValue(value);
  }, [value]);

  return (
    <div className="widget slider-field">
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <label style={{ fontSize: "11px" }}>{label}</label>
        <span style={{ fontSize: "11px" }}>{localValue}</span>
      </div>
      <input
        type="range"
        className="nodrag"
        min={min}
        max={max}
        value={localValue}
        onChange={(e) => {
          const val = Number(e.target.value);
          setLocalValue(val);
        }}
        onMouseUp={() => {
          onChange(localValue);
        }}
        // Also support touch devices
        onTouchEnd={() => {
          onChange(localValue);
        }}
        style={{ width: "100%" }}
      />
    </div>
  );
};
