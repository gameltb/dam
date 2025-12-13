import React from 'react';
import { Handle, Position, NodeProps } from 'reactflow';

type EntityNodeData = {
  entityId: string;
};

export function EntityNode({ data }: NodeProps<EntityNodeData>) {
  return (
    <div style={{ padding: 10, border: '1px solid #ddd', borderRadius: 5, background: '#f0f0f0' }}>
      <strong>Entity:</strong> {data.entityId}
      <Handle type="source" position={Position.Right} />
    </div>
  );
}
