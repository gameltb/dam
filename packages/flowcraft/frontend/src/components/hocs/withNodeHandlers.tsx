// src/components/hocs/withNodeHandlers.tsx

import React from "react";
import { type Node, type NodeProps } from "@xyflow/react";
import { BaseNode } from "../base/BaseNode";
import { Handle } from "../base/Handle";
import { Position } from "@xyflow/react";

export interface NodeRendererProps<T extends Node> {
  id: string;
  data: T["data"];
}

export function withNodeHandlers<
  T extends Node<{ outputType?: string }, string | undefined>,
>(
  renderMedia: (props: NodeRendererProps<T>) => React.ReactNode,
  renderWidgets: (
    props: NodeRendererProps<T>,
    onToggleMode: () => void,
  ) => React.ReactNode,
) {
  return function NodeWithHandlers(props: NodeProps<T>) {
    const { id } = props;

    return (
      <div data-testid={`${props.type}-node-${id}`}>
        <Handle type="target" position={Position.Left} />
        <BaseNode
          {...props}
          renderMedia={(data) => renderMedia({ id, data: data as T["data"] })}
          renderWidgets={(data, onToggleMode) =>
            renderWidgets({ id, data: data as T["data"] }, onToggleMode)
          }
        />
        <Handle type="source" position={Position.Right} />
      </div>
    );
  };
}
