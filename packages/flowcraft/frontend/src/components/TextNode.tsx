// src/components/TextNode.tsx

import {
  withNodeHandlers,
  type NodeRendererProps,
} from "./hocs/withNodeHandlers";
import { type Node } from "@xyflow/react";
import { TextField } from "./widgets/TextField";
import { WidgetWrapper } from "./widgets/WidgetWrapper";

export type TextNodeData = {
  label: string;
  onChange: (id: string, data: { label: string }) => void;
  outputType: "text";
  inputType: "any";
};

export type TextNodeType = Node<TextNodeData, "text">;

const renderWidgets = (
  { id, data }: NodeRendererProps<TextNodeType>,
  onToggleMode: () => void,
) => (
  <WidgetWrapper isSwitchable={false} onToggleMode={onToggleMode}>
    <TextField
      value={data.label}
      onChange={(label) => data.onChange(id, { label })}
      placeholder="Enter text..."
    />
  </WidgetWrapper>
);

export const TextNode = withNodeHandlers<TextNodeType>(
  () => null, // No media view for text nodes
  renderWidgets,
);
