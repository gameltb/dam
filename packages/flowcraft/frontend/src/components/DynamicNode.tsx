import React, { memo, useCallback } from "react";
import {
  type NodeProps,
  NodeResizer,
  type ResizeDragEvent,
} from "@xyflow/react";
import { type DynamicNodeType, type DynamicNodeData } from "../types";
import { WidgetContent } from "./nodes/WidgetContent";
import { MediaContent } from "./nodes/MediaContent";
import { ChatRenderer } from "./media/ChatRenderer";
import { BaseNode } from "./base/BaseNode";
import { useNodeHandlers } from "../hooks/useNodeHandlers";
import { useFlowStore } from "../store/flowStore";
import { create as createProto } from "@bufbuild/protobuf";
import { GraphMutationSchema } from "../generated/flowcraft/v1/core/service_pb";
import { PresentationSchema } from "../generated/flowcraft/v1/core/base_pb";
import { toProtoNodeData } from "../utils/protoAdapter";
import { NodeErrorBoundary } from "./base/NodeErrorBoundary";

interface LayoutProps {
  width?: number;
  height?: number;
  measured?: { width: number; height: number };
}

const RenderMedia: React.FC<{
  id: string;
  data: DynamicNodeData;
  onOverflowChange?: (o: "visible" | "hidden") => void;
  width?: number;
  height?: number;
  measured?: { width: number; height: number };
}> = (props) => {
  const { id, data, onOverflowChange } = props;
  const layoutProps = props as LayoutProps;

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

const RenderWidgets: React.FC<{
  id: string;
  data: DynamicNodeData;
  selected?: boolean;
  onToggleMode: () => void;
}> = (props) => {
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

export const DynamicNode = memo(
  ({
    id,
    data,
    selected,
    type,
    positionAbsoluteX,
    positionAbsoluteY,
    ...rest
  }: NodeProps<DynamicNodeType>) => {
    const {
      minHeight,
      minWidth,
      shouldLockAspectRatio,
      containerStyle,
      onChange: updateNodeData,
    } = useNodeHandlers(data, selected, positionAbsoluteX, positionAbsoluteY);

    const handleResizeEnd = useCallback(
      (_event: ResizeDragEvent, params: { width: number; height: number }) => {
        const { applyMutations } = useFlowStore.getState();
        const presentation = createProto(PresentationSchema, {
          width: params.width,
          height: params.height,
          position: { x: positionAbsoluteX, y: positionAbsoluteY },
          isInitialized: true,
        });

        applyMutations([
          createProto(GraphMutationSchema, {
            operation: {
              case: "updateNode",
              value: {
                id: id,
                presentation,
                data: toProtoNodeData(data),
              },
            },
          }),
        ]);
      },
      [id, data, positionAbsoluteX, positionAbsoluteY],
    );

    return (
      <div
        className="custom-node"
        style={{
          ...containerStyle,
          overflow: "visible", // Ensure floating panel is not clipped
        }}
      >
        <style>{`
          .custom-node:hover .react-flow__handle { opacity: 1 !important; }
      `}</style>

        <NodeResizer
          isVisible={selected}
          minWidth={minWidth}
          minHeight={minHeight}
          keepAspectRatio={shouldLockAspectRatio}
          onResizeEnd={handleResizeEnd}
          handleStyle={{
            width: 8,
            height: 8,
            borderRadius: "50%",
            background: "var(--primary-color)",
            border: "2px solid white",
          }}
        />

        <NodeErrorBoundary nodeId={id}>
          <BaseNode<DynamicNodeType>
            id={id}
            data={data}
            selected={selected}
            type={data.typeId ?? type}
            x={positionAbsoluteX}
            y={positionAbsoluteY}
            renderMedia={RenderMedia}
            renderWidgets={RenderWidgets}
            renderChat={ChatRenderer}
            updateNodeData={updateNodeData}
            handles={null}
            {...rest}
          />
        </NodeErrorBoundary>
      </div>
    );
  },
);

DynamicNode.displayName = "DynamicNode";
