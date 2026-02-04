import { memo } from "react";

import { RenderMode } from "@/generated/flowcraft/v1/core/node_pb";
import { type AppNode, type DynamicNodeData } from "@/types";

import { ChatRenderer } from "../../media/ChatRenderer";

export const ChatNodeImplementation = memo(
  ({ data, id, selected: _selected }: { data: DynamicNodeData; id: string; node: AppNode; selected?: boolean }) => {
    const isWidgetMode = data.activeMode === RenderMode.MODE_WIDGETS;

    return (
      <div className="flex flex-col h-full w-full overflow-hidden">
        <div className="flex-1 overflow-hidden flex flex-col relative">
          {isWidgetMode ? (
            <div className="flex-1 flex items-center justify-center p-4 opacity-30 text-[10px] uppercase tracking-widest bg-muted/10">
              Node Configuration
            </div>
          ) : (
            <ChatRenderer id={id} />
          )}
        </div>
      </div>
    );
  },
);
