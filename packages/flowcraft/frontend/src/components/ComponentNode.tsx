import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";

export type ComponentNodeData = {
  componentName: string;
  json: string;
};

export type ComponentNodeType = Node<ComponentNodeData, 'component'>;

export function ComponentNode({ data }: NodeProps<ComponentNodeType>) {
  return (
    <div
      style={{
        padding: 10,
        border: "1px solid #ddd",
        borderRadius: 5,
        background: "white",
        width: 300,
      }}
    >
      <Handle type="target" position={Position.Left} />
      <strong>{data.componentName}</strong>
      <pre
        style={{ margin: 0, whiteSpace: "pre-wrap", wordBreak: "break-all" }}
      >
        {data.json}
      </pre>
    </div>
  );
}
