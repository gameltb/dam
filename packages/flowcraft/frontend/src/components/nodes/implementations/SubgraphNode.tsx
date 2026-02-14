import { ArrowUpRight, Layers, MessageSquareText } from "lucide-react";
import { memo, useCallback } from "react";

import { RenderMode } from "@/generated/flowcraft/v1/core/node_pb";
import { useNavigation } from "@/hooks/graph/useNavigation";
import { type AppNode, type DynamicNodeData } from "@/types";

import { NodeHeaderSection } from "../sections/NodeHeaderSection";

export const SubgraphNodeImplementation = memo(
  ({ data, id, selected }: { data: DynamicNodeData; id: string; node: AppNode; selected?: boolean }) => {
    const isWidgetMode = data.activeMode === RenderMode.MODE_WIDGETS;
    const extensionValue = (data.extension?.value as any) || {};
    const { navigateTo } = useNavigation();

    const handleEnter = useCallback(() => {
      navigateTo(id, { focusNodeId: id });
    }, [id, navigateTo]);

    return (
      <div className="flex flex-col h-full w-full overflow-hidden">
        <NodeHeaderSection data={data} id={id} selected={selected} />
        <div className="flex-1 overflow-hidden relative h-full">
          {isWidgetMode ? (
            <div className="flex-1 flex items-center justify-center opacity-30 text-[10px] uppercase tracking-widest bg-muted/10 p-4">
              Subgraph Metadata
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center h-full w-full bg-primary/5 text-primary p-6 gap-3 group/subgraph">
              <div className="p-4 bg-primary/10 rounded-full transition-transform group-hover/subgraph:scale-110">
                <Layers className="text-primary opacity-80" size={32} />
              </div>
              <div className="flex flex-col items-center text-center">
                <span className="text-xs font-black uppercase tracking-widest opacity-60">Subgraph Session</span>
                <span className="text-[10px] font-mono opacity-40 break-all px-4">
                  {String(extensionValue.subgraphId || id)}
                </span>
              </div>
              <div className="flex items-center gap-1.5 px-2 py-1 bg-primary/10 rounded text-[9px] font-bold opacity-70">
                <MessageSquareText size={10} />
                {extensionValue.nodeCount ?? 0} NODES
              </div>

              <button
                className="mt-2 flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-full text-[10px] font-bold uppercase tracking-wider opacity-0 group-hover/subgraph:opacity-100 transition-all hover:scale-105 active:scale-95 shadow-lg"
                onClick={(e) => {
                  e.stopPropagation();
                  handleEnter();
                }}
              >
                Enter Subgraph
                <ArrowUpRight size={12} />
              </button>
            </div>
          )}
        </div>
      </div>
    );
  },
);
