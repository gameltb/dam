import React, { useState } from "react";
import { type Node } from "@xyflow/react";
import { type DynamicNodeData } from "../../types";

// Define the rendering modes
export type RenderMode = "media" | "widgets" | "markdown";

// Define the props for the BaseNode component

export interface BaseNodeProps<T extends Node> {
  id: string;
  data: T["data"];
  selected?: boolean;
  style?: React.CSSProperties;
  // The initial rendering mode for the node
  initialMode?: RenderMode;
  // Components to render
  renderMedia?: React.ComponentType<React.ComponentProps<any>>; // eslint-disable-line @typescript-eslint/no-explicit-any
  renderWidgets?: React.ComponentType<React.ComponentProps<any>>; // eslint-disable-line @typescript-eslint/no-explicit-any
  handles?: React.ReactNode;
  wrapperStyle?: React.CSSProperties;
  onOverflowChange?: (overflow: "visible" | "hidden") => void;
}

export function BaseNode<
  T extends Node<Record<string, unknown>, string | undefined>,
>({
  id,
  data,
  initialMode = "widgets",
  renderMedia: RenderMedia,
  renderWidgets: RenderWidgets,
  handles,
  wrapperStyle,
  onOverflowChange,
  ...rest
}: BaseNodeProps<T>) {
  const [internalMode, setInternalMode] = useState<RenderMode>(initialMode);
  const [overflow, setOverflow] = useState<"visible" | "hidden">("hidden");

  // Use data.activeMode if provided, otherwise fallback to internal state
  const mode = (data.activeMode as RenderMode) || internalMode;

  // Function to toggle the rendering mode
  const toggleMode = () => {
    const nextMode = mode === "widgets" ? "media" : "widgets";
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

  const isMedia = mode === "media";

  return (
    <>
      <div
        style={{
          width: "100%",
          height: "100%",
          padding: isMedia ? 0 : 12,
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
            data={data}
            {...rest}
            onOverflowChange={handleOverflowChange}
          />
        )}
        {!isMedia && RenderWidgets && (
          <RenderWidgets
            id={id}
            data={data}
            {...rest}
            onToggleMode={toggleMode}
          />
        )}
      </div>
      {handles}
    </>
  );
}
