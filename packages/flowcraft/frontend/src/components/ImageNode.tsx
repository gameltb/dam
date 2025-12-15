import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";

export type ImageNodeData = {
  url: string;
  onChange: (id: string, data: { url: string }) => void;
};

export type ImageNodeType = Node<ImageNodeData, "image">;

export function ImageNode({ id, data }: NodeProps<ImageNodeType>) {
  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    data.onChange(id, { url: event.target.value });
  };

  return (
    <div className="custom-node">
      <Handle type="target" position={Position.Left} />
      <input
        type="text"
        value={data.url}
        onChange={handleChange}
        placeholder="Image URL"
        style={{ width: "100%", marginBottom: 5, boxSizing: "border-box" }}
      />
      {data.url && (
        <img src={data.url} alt="Node content" style={{ maxWidth: "100%" }} />
      )}
      <Handle type="source" position={Position.Right} />
    </div>
  );
}
