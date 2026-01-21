import { type DescMessage } from "@bufbuild/protobuf";

import { ChatActionParamsSchema, ChatSyncBranchParamsSchema } from "@/generated/flowcraft/v1/actions/chat_actions_pb";
import {
  AddEdgeRequestSchema,
  AddNodeRequestSchema,
  AddSubGraphRequestSchema,
  ClearGraphRequestSchema,
  GraphMutationSchema,
  PathUpdateRequestSchema,
  RemoveEdgeRequestSchema,
  RemoveNodeRequestSchema,
  ReparentNodeRequestSchema,
} from "@/generated/flowcraft/v1/core/service_pb";

/**
 * 消息类型到 Schema 的核心映射表
 */
export const SCHEMA_MAP: Record<string, DescMessage> = {
  [AddEdgeRequestSchema.typeName]: AddEdgeRequestSchema,
  [AddNodeRequestSchema.typeName]: AddNodeRequestSchema,
  [AddSubGraphRequestSchema.typeName]: AddSubGraphRequestSchema,
  // 补全模板/动作参数 Schema
  [ChatActionParamsSchema.typeName]: ChatActionParamsSchema,
  [ChatSyncBranchParamsSchema.typeName]: ChatSyncBranchParamsSchema,
  [ClearGraphRequestSchema.typeName]: ClearGraphRequestSchema,
  [GraphMutationSchema.typeName]: GraphMutationSchema,
  [PathUpdateRequestSchema.typeName]: PathUpdateRequestSchema,
  [RemoveEdgeRequestSchema.typeName]: RemoveEdgeRequestSchema,

  [RemoveNodeRequestSchema.typeName]: RemoveNodeRequestSchema,
  [ReparentNodeRequestSchema.typeName]: ReparentNodeRequestSchema,
};

export function getSchemaForMessage(typeName: string): DescMessage | undefined {
  return SCHEMA_MAP[typeName];
}

/**
 * 根据模板 ID 获取 Schema
 */
export function getSchemaForTemplate(templateId: string): DescMessage | undefined {
  const templateMapping: Record<string, DescMessage> = {
    chat: ChatActionParamsSchema,
    chatSync: ChatSyncBranchParamsSchema,
  };
  return templateMapping[templateId];
}
