import { memo } from "react";

import { type AppNode, type DynamicNodeData } from "@/types";

import { NodeHeaderSection } from "../sections/NodeHeaderSection";

/**
 * Baseline implementation for nodes without a custom component.
 * Explicitly does NOT include RJSF.
 */
export const GenericNode = memo(
  ({ data, id, selected }: { data: DynamicNodeData; id: string; node: AppNode; selected?: boolean }) => {
    return (
      <div className="flex flex-col h-full w-full">
        <NodeHeaderSection data={data} id={id} selected={selected} />
        <div className="flex-1 flex items-center justify-center p-4">
          <div className="text-[10px] font-mono text-muted-foreground opacity-40 uppercase tracking-widest text-center">
            Generic Node implementation
            <br />
            (No specific UI defined)
          </div>
        </div>
      </div>
    );
  },
);

GenericNode.displayName = "GenericNode";
