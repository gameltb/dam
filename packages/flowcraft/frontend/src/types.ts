import type { TextNodeType } from "./components/TextNode";
import type { ImageNodeType } from "./components/ImageNode";
import type { EntityNodeType } from "./components/EntityNode";
import type { ComponentNodeType } from "./components/ComponentNode";

export type NodeData =
  | TextNodeType["data"]
  | ImageNodeType["data"]
  | EntityNodeType["data"]
  | ComponentNodeType["data"];

export interface TypedNodeData {
  inputType?: string;
  outputType?: string;
}

/* eslint-disable @typescript-eslint/no-explicit-any */
export function isTypedNodeData(data: any): data is TypedNodeData {
  return "inputType" in data || "outputType" in data;
}
/* eslint-enable @typescript-eslint/no-explicit-any */

export type AppNode =
  | TextNodeType
  | ImageNodeType
  | EntityNodeType
  | ComponentNodeType;
