import { create as createProto, fromJson } from "@bufbuild/protobuf";
import { ValueSchema } from "@bufbuild/protobuf/wkt";

import {
  AddNodeRequestSchema,
  PathUpdateRequest_UpdateType,
  PathUpdateRequestSchema,
  RemoveNodeRequestSchema,
  ReparentNodeRequestSchema,
} from "@/generated/flowcraft/v1/core/service_pb";
import { type MutationInput } from "@/store/types";
import { type AppNode } from "@/types";

import { appNodeToProto } from "./nodeProtoUtils";

/**
 * 根据当前状态计算给定指令的逆向指令
 */
export function calculateInverse(mutation: MutationInput, currentNodes: AppNode[]): MutationInput | null {
  switch (mutation.$typeName) {
    case AddNodeRequestSchema.typeName: {
      return createProto(RemoveNodeRequestSchema, { id: mutation.node?.nodeId });
    }

    case PathUpdateRequestSchema.typeName: {
      const node = currentNodes.find((n) => n.id === mutation.targetId);
      if (!node) return null;

      const pathParts = mutation.path.split(".");
      let oldValue: any = node;

      try {
        for (const part of pathParts) {
          if (part === "data") oldValue = node.data;
          else oldValue = oldValue[part];
        }
      } catch {
        oldValue = null;
      }

      return createProto(PathUpdateRequestSchema, {
        path: mutation.path,
        targetId: mutation.targetId,
        type: PathUpdateRequest_UpdateType.REPLACE,
        value: fromJson(ValueSchema, oldValue === undefined ? null : oldValue),
      });
    }

    case RemoveNodeRequestSchema.typeName: {
      const node = currentNodes.find((n) => n.id === mutation.id);
      if (!node) return null;
      return createProto(AddNodeRequestSchema, { node: appNodeToProto(node) });
    }

    case ReparentNodeRequestSchema.typeName: {
      const node = currentNodes.find((n) => n.id === mutation.nodeId);
      if (!node) return null;
      return createProto(ReparentNodeRequestSchema, {
        newParentId: node.parentId || "",
        newPosition: { x: node.position.x, y: node.position.y },
        nodeId: mutation.nodeId,
      });
    }
  }
  return null;
}

/**
 * 翻译指令为用户可读的描述
 */
export function getFriendlyDescription(mutation: MutationInput): string {
  switch (mutation.$typeName) {
    case AddNodeRequestSchema.typeName:
      return "新建节点";
    case PathUpdateRequestSchema.typeName:
      if (mutation.path.includes("displayName")) return "重命名节点";
      if (mutation.path.includes("position")) return "移动节点";
      if (mutation.path.includes("width") || mutation.path.includes("height")) return "调整节点大小";
      if (mutation.path.includes("activeMode")) return "切换显示模式";
      return `修改属性: ${mutation.path}`;
    case RemoveNodeRequestSchema.typeName:
      return "删除节点";
    case ReparentNodeRequestSchema.typeName:
      return "移动层级";
    default:
      return "未知操作";
  }
}
