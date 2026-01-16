import { create } from "@bufbuild/protobuf";
import { type NodeProps } from "@xyflow/react";
import { memo } from "react";

import { ChatNodeStateSchema } from "@/generated/flowcraft/v1/nodes/chat_node_pb";
import { useFlowStore } from "@/store/flowStore";
import { type DynamicNodeType } from "@/types";

import { ChatBot } from "./ChatBot";

export const ChatRenderer: React.FC<NodeProps<DynamicNodeType>> = memo(({ data, id }) => {
  const updateNodeData = useFlowStore((s) => s.updateNodeData);

  const headId = data.extension?.case === "chat" ? data.extension.value.conversationHeadId : "";
  const treeId = data.extension?.case === "chat" ? data.extension.value.treeId : "";

  const handleCreateBranch = () => {
    updateNodeData(id, {
      extension: {
        case: "chat",
        value: create(ChatNodeStateSchema, {
          conversationHeadId: headId,
          isHistoryCleared: false,
          treeId: treeId,
        }),
      },
    });
  };

  return (
    <div className="flex flex-col h-full w-full">
      <div className="p-2 border-b border-node-border flex justify-between items-center">
        <span className="text-xs font-bold uppercase opacity-50">{data.displayName || "Chat"}</span>
        <button
          className="text-[10px] bg-primary-color/10 text-primary-color px-2 py-1 rounded hover:bg-primary-color/20 transition-colors"
          onClick={handleCreateBranch}
        >
          New Branch
        </button>
      </div>
      <div className="flex-1 overflow-hidden min-h-[400px]">
        <ChatBot nodeId={id} />
      </div>
    </div>
  );
});
