import React from 'react';
import { Handle, Position, NodeProps } from 'reactflow';

type ComponentNodeData = {
  componentName: string;
  json: string;
};

export function ComponentNode({ data }: NodeProps<ComponentNodeData>) {
  return (
    <div style={{ padding: 10, border: '1px solid #ddd', borderRadius: 5, background: 'white', width: 300 }}>
      <Handle type="target" position={Position.Left} />
      <strong>{data.componentName}</strong>
      <pre style={{ margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
        {data.json}
      </pre>
    </div>
  );
}
