import React from "react";
import {
  withNodeHandlers,
  type NodeRendererProps,
} from "./hocs/withNodeHandlers";
import { type DynamicNodeType, type DynamicNodeData } from "../types";
import { WidgetContent } from "./nodes/WidgetContent";
import { MediaContent } from "./nodes/MediaContent";

interface LayoutProps {
  width?: number;
  height?: number;
  measured?: { width: number; height: number };
  style?: React.CSSProperties;
}

const RenderMedia: React.FC<
  NodeRendererProps<DynamicNodeType> & {
    onOverflowChange?: (o: "visible" | "hidden") => void;
  }
> = (props) => {
  const { id, data, onOverflowChange } = props;
  const layoutProps = props as unknown as LayoutProps;

  const nodeWidth = layoutProps.width ?? layoutProps.measured?.width ?? 240;

  const nodeHeight = layoutProps.height ?? layoutProps.measured?.height ?? 180;

  return (
    <MediaContent
      id={id}
      data={data}
      onOverflowChange={onOverflowChange}
      width={nodeWidth}
      height={nodeHeight}
    />
  );
};

const RenderWidgets: React.FC<
  NodeRendererProps<DynamicNodeType> & { onToggleMode: () => void }
> = (props) => {
  const { id, data, onToggleMode, selected } = props;

  return (
    <WidgetContent
      id={id}
      data={data}
      selected={selected}
      onToggleMode={onToggleMode}
    />
  );
};

export const DynamicNode = withNodeHandlers<DynamicNodeData, DynamicNodeType>(
  RenderMedia,
  RenderWidgets,
);
