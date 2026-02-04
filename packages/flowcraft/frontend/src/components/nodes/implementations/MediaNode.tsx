import { memo } from "react";

import { RenderMode } from "@/generated/flowcraft/v1/core/node_pb";
import { type AppNode, type DynamicNodeData } from "@/types";

import { NodeShell } from "../../base/NodeShell";
import { MediaContent } from "../MediaContent";

export const MediaNodeImplementation = memo(
  ({ data, id, node, selected }: { data: DynamicNodeData; id: string; node: AppNode; selected?: boolean }) => {
    const isWidgetMode = data.activeMode === RenderMode.MODE_WIDGETS;

    const aspectRatio = data.media?.aspectRatio || 1.33;

    return (
      <div className="flex flex-col h-full w-full overflow-hidden">
        <NodeShell aspectRatio={isWidgetMode ? undefined : aspectRatio} isNested={true} nodeId={id} selected={selected}>
          <div className="flex-1 overflow-hidden relative">
            {isWidgetMode ? (
              <div className="flex-1 flex items-center justify-center opacity-30 text-[10px] uppercase tracking-widest bg-muted/10 p-4 h-full">
                Display Settings
              </div>
            ) : (
              <MediaContent data={data} height={node.measured?.height} id={id} width={node.measured?.width} />
            )}
          </div>
        </NodeShell>
      </div>
    );
  },
);
