import { Send, Settings2 } from "lucide-react";
import { memo } from "react";

import { useSpacetimeChat } from "@/hooks/useSpacetimeChat";
import { useFlowStore } from "@/store/flowStore";
import { type DynamicNodeData } from "@/types";

import { withScopeLens } from "../hocs/withScopeLens";
import { Button } from "../ui/button";
import { ChatBot } from "./ChatBot";

const ChatRendererComponent = ({ id }: { id: string }) => {
  const node = useFlowStore((s) => s.allNodes.find((n) => n.id === id));
  const data = node?.data as DynamicNodeData | undefined;
  const treeId = data?.extension?.case === "chat" ? data.extension.value.treeId : undefined;
  const { messages } = useSpacetimeChat(treeId || id);

  return (
    <div className="flex flex-col h-full bg-background/50">
      <div className="flex items-center justify-between px-3 py-2 border-b border-node-border bg-muted/20">
        <div className="flex items-center gap-2">
          <Send className="text-primary" size={14} />
          <span className="text-[10px] font-bold uppercase">Chat Lens</span>
        </div>
        <div className="flex gap-1">
          <Button className="h-6 w-6" size="icon" variant="ghost">
            <Settings2 size={14} />
          </Button>
        </div>
      </div>

      <div className="flex-1 overflow-hidden relative">
        <ChatBot nodeId={id} />
      </div>

      <div className="px-3 py-1 border-t border-node-border bg-muted/10">
        <span className="text-[8px] text-muted-foreground uppercase">{messages.length} Active context nodes</span>
      </div>
    </div>
  );
};

export const ChatRenderer = Object.assign(memo(withScopeLens(ChatRendererComponent)), {
  minSize: { height: 400, width: 350 },
});
