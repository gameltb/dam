import { type Node, type NodeProps, NodeResizer } from "@xyflow/react";
import { memo } from "react";

import { useUiStore } from "@/store/uiStore";
import { AppNodeType } from "@/types";

import { NodeShell } from "../base/NodeShell";
import { NodeLabel } from "./NodeLabel";
import { SubGraphActions } from "./parts/SubGraphActions";

export type GroupNodeData = Record<string, unknown> & {
  displayName?: string;
};

export type GroupNodeType = Node<GroupNodeData, AppNodeType.GROUP>;

export const GroupNode = memo(({ data: _data, id, selected }: NodeProps<GroupNodeType>) => {
  const navigatingNodeId = useUiStore((s) => s.navigatingNodeId);
  const setNavigatingNode = useUiStore((s) => s.setNavigatingNode);
  const resetNavigatingNode = useUiStore((s) => s.resetNavigatingNode);
  const isHovered = navigatingNodeId === id;

  return (
    <div
      className="group/group-node relative h-full w-full"
      onMouseEnter={(e) => {
        setNavigatingNode(id);
        useUiStore.getState().setLastMousePos({ x: e.clientX, y: e.clientY });
      }}
      onMouseLeave={() => {
        resetNavigatingNode(id);
      }}
    >
      <SubGraphActions id={id} isHovered={isHovered} />

      <NodeShell
        className="border-primary/20 bg-primary/5"
        nodeId={id}
        selected={selected}
        style={{
          borderRadius: "12px",
          borderStyle: "dashed",
          height: "100%",
          width: "100%",
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
          minHeight={100}
          minWidth={150}
        />

        <div className="flex flex-col h-full w-full overflow-hidden">
          <NodeLabel selected={selected} />
          <div className="flex-1" />
        </div>
      </NodeShell>
    </div>
  );
});

GroupNode.displayName = "GroupNode";
