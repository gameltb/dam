import React, { useState } from "react";
import { type Node } from "@xyflow/react";
import { flowcraft_proto } from "../../generated/flowcraft_proto";
import { type DynamicNodeData } from "../../types";

const RenderMode = flowcraft_proto.v1.RenderMode;

export interface BaseNodeProps<T extends Node> {
  id: string;
  data: T["data"];
  selected?: boolean;
  style?: React.CSSProperties;
  initialMode?: flowcraft_proto.v1.RenderMode;
  renderMedia?: React.ComponentType<any>; // eslint-disable-line @typescript-eslint/no-explicit-any
  renderWidgets?: React.ComponentType<any>; // eslint-disable-line @typescript-eslint/no-explicit-any
  handles?: React.ReactNode;
  wrapperStyle?: React.CSSProperties;
  onOverflowChange?: (overflow: "visible" | "hidden") => void;
}

export function BaseNode<T extends Node>({
  id,
  data,
  initialMode = RenderMode.MODE_WIDGETS,
  renderMedia: RenderMedia,
  renderWidgets: RenderWidgets,
  handles,
  wrapperStyle,
  onOverflowChange,
  ...rest
}: BaseNodeProps<T>) {
  const [internalMode, setInternalMode] =
    useState<flowcraft_proto.v1.RenderMode>(initialMode);
  // Default to visible so handles are never cut off
  const [overflow, setOverflow] = useState<"visible" | "hidden">("visible");

  const mode =
    (data.activeMode as flowcraft_proto.v1.RenderMode | undefined) ??
    internalMode;

  const toggleMode = () => {
    const nextMode =
      mode === RenderMode.MODE_WIDGETS
        ? RenderMode.MODE_MEDIA
        : RenderMode.MODE_WIDGETS;
    const dynamicData = data as unknown as DynamicNodeData;
    if (typeof dynamicData.onChange === "function") {
      dynamicData.onChange(id, { activeMode: nextMode });
    } else {
      setInternalMode(nextMode);
    }
  };

  const handleOverflowChange = (newOverflow: "visible" | "hidden") => {
    setOverflow(newOverflow);
    onOverflowChange?.(newOverflow);
  };

  const isMedia = mode === RenderMode.MODE_MEDIA;

  return (
    <>
      <div
        style={{
          width: "100%",
          height: "100%",
          // Remove horizontal padding from here, move it to sub-components
          paddingTop: isMedia ? 0 : 0,
          paddingBottom: isMedia ? 0 : 0,
          overflow: overflow,
          display: "flex",
          flexDirection: "column",
          boxSizing: "border-box",
          borderRadius: "inherit",
          ...wrapperStyle,
        }}
      >
        {isMedia && RenderMedia && (
          <RenderMedia
            id={id}
            data={data as DynamicNodeData}
            {...rest}
            onOverflowChange={handleOverflowChange}
          />
        )}
        {!isMedia && RenderWidgets && (
          <RenderWidgets
            id={id}
            data={data as DynamicNodeData}
            {...rest}
            onToggleMode={toggleMode}
          />
        )}
      </div>
      {handles}
    </>
  );
}
