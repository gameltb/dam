import { create as createProto } from "@bufbuild/protobuf";
import {
  type NodeProps,
  NodeResizer,
  type ResizeDragEvent,
} from "@xyflow/react";
import { AlertCircle } from "lucide-react";
import React, { memo, useCallback } from "react";

import { PresentationSchema } from "@/generated/flowcraft/v1/core/base_pb";
import { GraphMutationSchema } from "@/generated/flowcraft/v1/core/service_pb";
import { useNodeHandlers } from "@/hooks/useNodeHandlers";
import { cn } from "@/lib/utils";
import { useFlowStore } from "@/store/flowStore";
import { useTaskStore } from "@/store/taskStore";
import {
  type DynamicNodeData,
  type DynamicNodeType,
  OverflowMode,
  TaskStatus,
} from "@/types";
import { toProtoNodeData } from "@/utils/protoAdapter";

import { BaseNode } from "./base/BaseNode";
import { NodeErrorBoundary } from "./base/NodeErrorBoundary";
import { ChatRenderer } from "./media/ChatRenderer";
import { MediaContent } from "./nodes/MediaContent";
import { WidgetContent } from "./nodes/WidgetContent";

interface LayoutProps {
  height?: number;
  measured?: { height: number; width: number };
  width?: number;
}

const RenderMedia: React.FC<{
  data: DynamicNodeData;
  height?: number;
  id: string;
  measured?: { height: number; width: number };
  onOverflowChange?: (o: OverflowMode) => void;
  width?: number;
}> = (props) => {
  const { data, id, onOverflowChange } = props;
  const layoutProps = props as LayoutProps;

  const nodeWidth = layoutProps.width ?? layoutProps.measured?.width ?? 240;
  const nodeHeight = layoutProps.height ?? layoutProps.measured?.height ?? 180;

  return (
    <MediaContent
      data={data}
      height={nodeHeight}
      id={id}
      onOverflowChange={onOverflowChange}
      width={nodeWidth}
    />
  );
};

const RenderWidgets: React.FC<{
  data: DynamicNodeData;
  id: string;
  onToggleMode: () => void;
  selected?: boolean;
}> = (props) => {
  const { data, id, onToggleMode, selected } = props;

  return (
    <WidgetContent
      data={data}
      id={id}
      onToggleMode={onToggleMode}
      selected={selected}
    />
  );
};

export const DynamicNode = memo(
  ({
    data,
    id,
    positionAbsoluteX,
    positionAbsoluteY,
    selected,
    type,
    ...rest
  }: NodeProps<DynamicNodeType>) => {
    const {
      containerStyle,
      minHeight,
      minWidth,
      onChange: updateNodeData,
      shouldLockAspectRatio,
    } = useNodeHandlers(data, selected, positionAbsoluteX, positionAbsoluteY);

    const hasError = useTaskStore((s) =>
      Object.values(s.tasks).some(
        (t) => t.nodeId === id && t.status === TaskStatus.TASK_FAILED,
      ),
    );

    const handleResizeEnd = useCallback(
      (_event: ResizeDragEvent, params: { height: number; width: number }) => {
        const { applyMutations } = useFlowStore.getState();
        const presentation = createProto(PresentationSchema, {
          height: params.height,
          isInitialized: true,
          position: { x: positionAbsoluteX, y: positionAbsoluteY },
          width: params.width,
        });

        applyMutations([
          createProto(GraphMutationSchema, {
            operation: {
              case: "updateNode",
              value: {
                data: toProtoNodeData(data),
                id: id,
                presentation,
              },
            },
          }),
        ]);
      },
      [id, data, positionAbsoluteX, positionAbsoluteY],
    );

    return (
      <div
        className={cn(
          "custom-node relative transition-shadow duration-300",
          hasError &&
            "ring-2 ring-destructive ring-offset-2 ring-offset-background",
        )}
        style={{
          ...containerStyle,
          overflow: "visible", // Ensure floating panel is not clipped
        }}
      >
        <style>{`
          .custom-node:hover .react-flow__handle { opacity: 1 !important; }
      `}</style>

        {hasError && !selected && (
          <div className="absolute -top-3 -right-3 bg-destructive text-white rounded-full p-1.5 shadow-xl animate-bounce z-[1000]">
            <AlertCircle size={14} />
          </div>
        )}

        <NodeResizer
          handleStyle={{
            background: "var(--primary-color)",
            border: "2px solid white",
            borderRadius: "50%",
            height: 8,
            width: 8,
          }}
          isVisible={selected}
          keepAspectRatio={shouldLockAspectRatio}
          minHeight={minHeight}
          minWidth={minWidth}
          onResizeEnd={handleResizeEnd}
        />

        <NodeErrorBoundary nodeId={id}>
          <BaseNode<DynamicNodeType>
            data={data}
            handles={null}
            id={id}
            renderChat={ChatRenderer}
            renderMedia={RenderMedia}
            renderWidgets={RenderWidgets}
            selected={selected}
            type={data.typeId ?? type}
            updateNodeData={updateNodeData}
            x={positionAbsoluteX}
            y={positionAbsoluteY}
            {...rest}
          />
        </NodeErrorBoundary>
      </div>
    );
  },
);

DynamicNode.displayName = "DynamicNode";
