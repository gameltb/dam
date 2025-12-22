import { useState } from "react";
import type { Node, NodeProps } from "@xyflow/react";

// Define the rendering modes
export type RenderMode = "media" | "widgets";

// Define the props for the BaseNode component
export interface BaseNodeProps<T extends Node> extends NodeProps<T> {
  // The initial rendering mode for the node
  initialMode?: RenderMode;
  // A function that renders the media view of the node
  renderMedia?: (data: T["data"]) => React.ReactNode;
  // A function that renders the widgets view of the node
  renderWidgets?: (
    data: T["data"],
    onToggleMode: () => void,
  ) => React.ReactNode;
}

export function BaseNode<
  T extends Node<Record<string, unknown>, string | undefined>,
>({
  data,
  initialMode = "widgets",
  renderMedia,
  renderWidgets,
}: BaseNodeProps<T>) {
  const [mode, setMode] = useState<RenderMode>(initialMode);

  // Function to toggle the rendering mode
  const toggleMode = () => {
    setMode((prevMode) => (prevMode === "widgets" ? "media" : "widgets"));
  };

  // Determine which view to render based on the current mode
  const content =
    mode === "media" && renderMedia
      ? renderMedia(data)
      : renderWidgets
        ? renderWidgets(data, toggleMode)
        : null;

  return <div className="base-node">{content}</div>;
}
