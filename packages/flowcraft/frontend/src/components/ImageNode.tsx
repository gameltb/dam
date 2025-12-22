// src/components/ImageNode.tsx

import {
  withNodeHandlers,
  type NodeRendererProps,
} from "./hocs/withNodeHandlers";
import { type Node } from "@xyflow/react";
import { ImageRenderer } from "./media/ImageRenderer";
import { TextField } from "./widgets/TextField";
import { WidgetWrapper } from "./widgets/WidgetWrapper";

export type ImageNodeData = {
  url: string;
  onChange: (id: string, data: { url: string }) => void;
  outputType: "image";
  inputType: "any";
};

export type ImageNodeType = Node<ImageNodeData, "image">;

const renderMedia = ({ data }: NodeRendererProps<ImageNodeType>) => (
  <ImageRenderer url={data.url} />
);

const renderWidgets = (
  { id, data }: NodeRendererProps<ImageNodeType>,
  onToggleMode: () => void,
) => (
  <WidgetWrapper isSwitchable={true} onToggleMode={onToggleMode}>
    <TextField
      value={data.url}
      onChange={(url) => data.onChange(id, { url })}
      placeholder="Image URL"
    />
  </WidgetWrapper>
);

export const ImageNode = withNodeHandlers<ImageNodeType>(
  renderMedia,
  renderWidgets,
);
