import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";

export type EntityNodeData = {
  entityId: string;
};

export type EntityNodeType = Node<EntityNodeData, "entity">;

export function EntityNode({ data }: NodeProps<EntityNodeType>) {
  return (
    <div className="custom-node">
      <strong>Entity:</strong> {data.entityId}
      <Handle type="source" position={Position.Right} />
    </div>
  );
}
