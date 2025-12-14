import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";

export type EntityNodeData = {
  entityId: string;
};

export type EntityNodeType = Node<EntityNodeData, 'entity'>;

export function EntityNode({ data }: NodeProps<EntityNodeType>) {
  return (
    <div
      style={{
        padding: 10,
        border: "1px solid #ddd",
        borderRadius: 5,
        background: "#f0f0f0",
      }}
    >
      <strong>Entity:</strong> {data.entityId}
      <Handle type="source" position={Position.Right} />
    </div>
  );
}
