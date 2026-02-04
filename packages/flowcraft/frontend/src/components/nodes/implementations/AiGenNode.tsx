import { memo } from "react";

import { RenderMode } from "@/generated/flowcraft/v1/core/node_pb";
import { type AppNode, type DynamicNodeData } from "@/types";

import { NodeHeaderSection } from "../sections/NodeHeaderSection";

export const AiGenNodeImplementation = memo(
  ({ data, id, selected }: { data: DynamicNodeData; id: string; node: AppNode; selected?: boolean }) => {
    const isWidgetMode = data.activeMode === RenderMode.MODE_WIDGETS;

    return (
      <div className="flex flex-col h-full w-full overflow-hidden">
        <NodeHeaderSection data={data} id={id} selected={selected} />
        <div className="flex-1 overflow-hidden relative">
          {isWidgetMode ? (
            <div className="flex-1 flex items-center justify-center opacity-30 text-[10px] uppercase tracking-widest bg-muted/10 p-4">
              AI Parameters
            </div>
          ) : (
            <div className="p-4 flex flex-col h-full items-center justify-center bg-primary/5 rounded-md border border-primary/20 m-2">
              <div className="text-[10px] font-black uppercase tracking-widest text-primary mb-2">
                AI Generation View
              </div>
              <div className="flex-1 w-full bg-background/50 p-3 rounded border border-border/50 overflow-auto">
                <pre className="text-[10px] font-mono opacity-80 whitespace-pre-wrap">
                  {JSON.stringify(data.extension?.value || {}, null, 2)}
                </pre>
              </div>
            </div>
          )}
        </div>
      </div>
    );
  },
);
