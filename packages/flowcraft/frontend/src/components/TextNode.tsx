import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";

export type TextNodeData = {
  label: string;
  onChange: (id: string, data: { label: string }) => void;
};

export type TextNodeType = Node<TextNodeData, "text">;

export function TextNode({ id, data }: NodeProps<TextNodeType>) {
  const handleChange = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
    data.onChange(id, { label: event.target.value });
  };

  return (
    <div className="custom-node">
      <Handle type="target" position={Position.Left} />
      <textarea
        value={data.label}
        onChange={handleChange}
        style={{
          width: "100%",
          border: "none",
          background: "transparent",
          resize: "none",
        }}
        rows={3}
      />
      <Handle type="source" position={Position.Right} />
    </div>
  );
}
