import { useState } from "react";
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
  renderMedia?: React.ComponentType<{
    id: string;
    data: T["data"];
    [key: string]: unknown;
  }>;
  renderWidgets?: React.ComponentType<{
    id: string;
    data: T["data"];
    onToggleMode: () => void;
    [key: string]: unknown;
  }>;
}

export function BaseNode<
  T extends Node<Record<string, unknown>, string | undefined>,
>({
  id,
  data,
  initialMode = "widgets",
  renderMedia: RenderMedia,
  renderWidgets: RenderWidgets,
  ...rest
}: BaseNodeProps<T>) {
  const [internalMode, setInternalMode] = useState<RenderMode>(initialMode);

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

  const isMedia = mode === "media";

  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        padding: isMedia ? 0 : 12,
        overflow: "visible",
        display: "flex",
        flexDirection: "column",
        boxSizing: "border-box",
        borderRadius: "inherit",
      }}
    >
      {isMedia && RenderMedia && <RenderMedia id={id} data={data} {...rest} />}
      {!isMedia && RenderWidgets && (
        <RenderWidgets
          id={id}
          data={data}
          {...rest}
          onToggleMode={toggleMode}
        />
      )}
    </div>
  );
}
