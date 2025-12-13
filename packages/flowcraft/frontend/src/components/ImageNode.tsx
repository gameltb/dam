import React from 'react';
import { Handle, Position, NodeProps } from 'reactflow';

type ImageNodeData = {
  url: string;
  onChange: (id: string, data: { url: string }) => void;
};

export function ImageNode({ id, data }: NodeProps<ImageNodeData>) {
  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    data.onChange(id, { url: event.target.value });
  };

  return (
    <div style={{ padding: 10, border: '1px solid #ddd', borderRadius: 5, background: 'white' }}>
      <Handle type="target" position={Position.Left} />
      <input
        type="text"
        value={data.url}
        onChange={handleChange}
        placeholder="Image URL"
        style={{ width: '100%', marginBottom: 5, boxSizing: 'border-box' }}
      />
      {data.url && <img src={data.url} alt="Node content" style={{ maxWidth: '100%' }} />}
      <Handle type="source" position={Position.Right} />
    </div>
  );
}
