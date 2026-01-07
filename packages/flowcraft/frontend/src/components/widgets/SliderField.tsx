import React, { useEffect, useState } from "react";

export interface SliderFieldProps {
  label: string;
  max?: number;
  min?: number;
  onChange: (value: number) => void;
  value: number;
}

export const SliderField: React.FC<SliderFieldProps> = ({
  label,
  max = 100,
  min = 0,
  onChange,
  value,
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
          alignItems: "center",
          display: "flex",
          justifyContent: "space-between",
        }}
      >
        <label style={{ fontSize: "11px" }}>{label}</label>
        <span style={{ fontSize: "11px" }}>{localValue}</span>
      </div>
      <input
        className="nodrag"
        max={max}
        min={min}
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
        type="range"
        value={localValue}
      />
    </div>
  );
};
