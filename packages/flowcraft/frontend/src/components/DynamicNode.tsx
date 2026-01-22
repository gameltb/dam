import { type NodeProps, NodeResizer } from "@xyflow/react";
import { memo, useCallback, useMemo } from "react";
import { useShallow } from "zustand/react/shallow";

import { MediaType } from "@/generated/flowcraft/v1/core/base_pb";
import { TaskStatus } from "@/generated/flowcraft/v1/core/kernel_pb";
import { RenderMode } from "@/generated/flowcraft/v1/core/node_pb";
import { useNodeHandlers } from "@/hooks/useNodeHandlers";
import { useFlowStore } from "@/store/flowStore";
import { useTaskStore } from "@/store/taskStore";
import { useUiStore } from "@/store/uiStore";
import { type DynamicNodeType } from "@/types";

import { BaseNode } from "./base/BaseNode";
import { NodeErrorBoundary } from "./base/NodeErrorBoundary";
import { ChatRenderer } from "./media/ChatRenderer";
import { MEDIA_CONFIGS } from "./media/mediaConfigs";
import { MediaContent } from "./nodes/MediaContent";
import { WidgetContent } from "./nodes/WidgetContent";

/**
 * Maps incoming render mode to standardized RenderMode enum.
 */
const mapToRenderMode = (mode: number | RenderMode | { tag: string }, nodeId: string): RenderMode => {
  let resolvedMode: RenderMode = RenderMode.MODE_UNSPECIFIED;

  if (typeof mode === "object" && "tag" in mode) {
    resolvedMode = RenderMode[mode.tag as keyof typeof RenderMode] ?? RenderMode.MODE_UNSPECIFIED;
  } else if (typeof mode === "number") {
    resolvedMode = mode;
  } else if (typeof mode === "string") {
    resolvedMode = (RenderMode as any)[mode] ?? RenderMode.MODE_UNSPECIFIED;
  }

  if (resolvedMode === RenderMode.MODE_UNSPECIFIED) {
    throw new Error(`[DynamicNode] Node ${nodeId} has RenderMode.MODE_UNSPECIFIED. 
      Nodes must always have a concrete active mode.`);
  }

  return resolvedMode;
};

const ContentRenderer: React.FC<{
  data: DynamicNodeType["data"];
  id: string;
  onToggleMode: () => void;
  selected?: boolean;
}> = memo((props) => {
  const { data, id, onToggleMode, selected } = props;
  const mode = mapToRenderMode(data.activeMode, id);

  return (
    <NodeErrorBoundary nodeId={id}>
      <div className="w-full h-full overflow-hidden rounded-[inherit] flex flex-col">
        {(() => {
          switch (mode) {
            case RenderMode.MODE_CHAT:
              return <ChatRenderer data={data} id={id} />;
            case RenderMode.MODE_MEDIA:
              return <MediaContent data={data} id={id} />;
            case RenderMode.MODE_WIDGETS:
              return <WidgetContent data={data} id={id} onToggleMode={onToggleMode} selected={selected} />;
            default:
              throw new Error(`[DynamicNode] Unhandled RenderMode: ${RenderMode[mode] || mode} for node ${id}`);
          }
        })()}
      </div>
    </NodeErrorBoundary>
  );
});

export const DynamicNode = memo(
  ({ data, id, positionAbsoluteX, positionAbsoluteY, selected }: NodeProps<DynamicNodeType>) => {
    const { containerStyle, shouldLockAspectRatio } = useNodeHandlers(
      data,
      selected,
      positionAbsoluteX,
      positionAbsoluteY,
    );

    const activeScopeId = useUiStore((s) => s.activeScopeId);
    const isActiveScope = activeScopeId === id;
    const { allNodes, nodeDraft } = useFlowStore(
      useShallow((s) => ({
        allNodes: s.allNodes,
        nodeDraft: s.nodeDraft,
      })),
    );

    // Calculate effective constraints from registry
    const { minHeight, minWidth } = useMemo(() => {
      const mode = mapToRenderMode(data.activeMode, id);
      const constraints = MEDIA_CONFIGS[mode as number] || { minHeight: 150, minWidth: 200 };
      const modeSpecific = (constraints as any).modeConstraints?.[mode as number];

      if (modeSpecific) {
        return { minHeight: modeSpecific.minHeight, minWidth: modeSpecific.minWidth };
      }

      // Special handling for Media mode which has its own registry (mediaConfigs)
      if (mode === RenderMode.MODE_MEDIA) {
        const mediaType =
          data.media?.type ?? (data.extension?.case === "visual" ? (data.extension.value as any).type : undefined);
        const size = MediaContent.getMinSize(mediaType ?? MediaType.MEDIA_UNSPECIFIED);
        return { minHeight: size.height, minWidth: size.width };
      }

      return {
        minHeight: (constraints as any).minHeight ?? 150,
        minWidth: (constraints as any).minWidth ?? 200,
      };
    }, [data.activeMode, data.templateId, data.media?.type, data.extension, id]);

    const hasError = useTaskStore((s) =>
      Object.values(s.tasks).some((t) => t.nodeId === id && t.status === TaskStatus.FAILED),
    );

    const onToggleMode = useCallback(() => {
      const node = allNodes.find((n) => n.id === id);
      if (!node) return;

      const mode = mapToRenderMode(data.activeMode, id);
      const nextMode = mode === RenderMode.MODE_WIDGETS ? RenderMode.MODE_MEDIA : RenderMode.MODE_WIDGETS;

      // Use ORM-style update with Result handling
      const res = nodeDraft(node);
      if (res.ok) {
        res.value.data.activeMode = nextMode;
      }
    }, [allNodes, data.activeMode, id, nodeDraft]);

    const handleResizeEnd = useCallback(
      (_: any, params: { height: number; width: number }) => {
        const node = allNodes.find((n) => n.id === id);
        if (!node) return;

        const res = nodeDraft(node);
        if (res.ok) {
          const draft = res.value;
          draft.presentation.width = params.width;
          draft.presentation.height = params.height;
        }
      },
      [allNodes, id, nodeDraft],
    );

    return (
      <BaseNode
        className={hasError ? "border-destructive shadow-[0_0_15px_rgba(239,68,68,0.3)]" : ""}
        style={{
          ...containerStyle,
          borderStyle: isActiveScope ? "dashed" : "solid",
          opacity: isActiveScope ? 0.3 : 1,
        }}
      >
        <NodeResizer
          color="var(--primary-color)"
          handleStyle={{
            backgroundColor: "var(--primary-color)",
            border: "2px solid white",
            borderRadius: "50%",
            height: 10,
            width: 10,
          }}
          isVisible={selected}
          keepAspectRatio={shouldLockAspectRatio}
          minHeight={minHeight}
          minWidth={minWidth}
          onResizeEnd={handleResizeEnd}
        />

        {isActiveScope ? (
          <div className="w-full h-full flex items-center justify-center pointer-events-none select-none">
            <span className="text-2xl font-black text-primary/20 uppercase tracking-widest">{data.displayName}</span>
          </div>
        ) : (
          <ContentRenderer data={data} id={id} onToggleMode={onToggleMode} selected={selected} />
        )}
      </BaseNode>
    );
  },
);
