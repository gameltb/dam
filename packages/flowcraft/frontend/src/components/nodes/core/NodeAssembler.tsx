import { type NodeProps, NodeResizer } from "@xyflow/react";
import { memo, useCallback, useMemo } from "react";
import { useShallow } from "zustand/react/shallow";

import { RenderMode } from "@/generated/flowcraft/v1/core/node_pb";
import { SizingStrategy } from "@/hooks/nodes/useNodeDimensionManager";
import { useNodeHandlers } from "@/hooks/nodes/useNodeHandlers";
import { useNodeMutation } from "@/hooks/nodes/useNodeMutation";
import { useFlowStore } from "@/store/flowStore";
import { useUiStore } from "@/store/uiStore";
import { type DynamicNodeType } from "@/types";
import { RESIZER_COLOR, RESIZER_HANDLE_STYLE } from "@/utils/themeUtils";

import { NodeShell } from "../../base/NodeShell";
import { NodeJumpOverlay } from "../parts/NodeJumpOverlay";
import { GenericNode } from "./GenericNode";
import { resolveNodeComponent } from "./NodeRegistry";

/**
 * NodeAssembler
 * Universal entry point that coordinates Shell, Resizer, and Content.
 */
export const NodeAssembler = memo((props: NodeProps<DynamicNodeType>) => {
  const { data, id, selected } = props;

  // 1. Surgical Store Access
  // We only subscribe to what determines the core strategy or identity.
  const { exists, presentationHeight } = useFlowStore(
    useShallow((s) => {
      const n = s.nodesById[id];
      return {
        exists: !!n,
        presentationHeight: n?.presentation?.height,
      };
    }),
  );

  const setNavigatingNode = useUiStore((s) => s.setNavigatingNode);
  const resetNavigatingNode = useUiStore((s) => s.resetNavigatingNode);
  const { updateLayout } = useNodeMutation(id);

  const { containerStyle, shouldLockAspectRatio } = useNodeHandlers(data, selected);

  // 2. Implementation Resolution
  const extension = data.extension as undefined | { case: string };
  const extensionCase = extension?.case;
  const implementation = useMemo(() => {
    return extensionCase ? resolveNodeComponent(data) : undefined;
  }, [extensionCase, data]);

  const isGeneric = !implementation;
  const isWidgetMode = data.activeMode === RenderMode.MODE_WIDGETS;
  const hasExplicitHeight = Number.isFinite(presentationHeight);

  const ContentComponent = implementation?.component ?? GenericNode;

  // 3. Dimension Management Strategy
  const sizingStrategy = useMemo(() => {
    if (extensionCase === "visual" || extensionCase === "acoustic") {
      return SizingStrategy.ASPECT_RATIO;
    }
    if ((isGeneric || isWidgetMode) && !hasExplicitHeight) {
      return SizingStrategy.CONTENT_FIT;
    }
    return SizingStrategy.MANUAL;
  }, [extensionCase, isGeneric, isWidgetMode, hasExplicitHeight]);

  // Handle resizing end by updating both local and persistence state
  const handleResizeEnd = useCallback(
    (_: unknown, params: { height: number; width: number }) => {
      updateLayout({ height: params.height, width: params.width });
    },
    [updateLayout],
  );

  const minConstraints = implementation?.constraints ?? { minHeight: 100, minWidth: 200 };

  // 4. Interaction Handlers
  const onMouseEnter = useCallback(
    (e: React.MouseEvent) => {
      setNavigatingNode(id, false);
      // Update mouse pos for potential context-aware actions
      useUiStore.getState().setLastMousePos({ x: e.clientX, y: e.clientY });
    },
    [id, setNavigatingNode],
  );

  const onMouseLeave = useCallback(() => {
    resetNavigatingNode(id);
  }, [id, resetNavigatingNode]);

  if (!exists) return null;

  return (
    <div className="group/node relative w-full h-full" onMouseEnter={onMouseEnter} onMouseLeave={onMouseLeave}>
      <NodeJumpOverlay id={id} />

      <NodeShell nodeId={id} selected={selected} sizingStrategy={sizingStrategy} style={containerStyle}>
        <NodeResizer
          color={RESIZER_COLOR}
          handleStyle={RESIZER_HANDLE_STYLE}
          isVisible={selected}
          keepAspectRatio={shouldLockAspectRatio}
          minHeight={minConstraints.minHeight}
          minWidth={minConstraints.minWidth}
          onResizeEnd={handleResizeEnd}
        />

        <div className="w-full h-full overflow-hidden rounded-[inherit] flex flex-col">
          {/* Injecting content based on resolved type */}
          <ContentComponent data={data} id={id} node={props as never} selected={selected} />
        </div>
      </NodeShell>
    </div>
  );
});

NodeAssembler.displayName = "NodeAssembler";
