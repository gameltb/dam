import React from "react";

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
        <span style={{ fontSize: "11px" }}>{value}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{ width: "100%" }}
      />
    </div>
  );
};
