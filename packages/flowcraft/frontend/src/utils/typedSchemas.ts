import { type DescMessage } from "@bufbuild/protobuf";
import { type RJSFSchema } from "@rjsf/utils";

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
import ChatActionSchema from "@/generated/schemas/flowcraft.v1.actions.ChatActionParams.schema.json";
// Action Parameter JSON Schemas (Only for RJSF)
import ImageEnhanceSchema from "@/generated/schemas/flowcraft_proto.v1.ImageEnhanceParams.schema.json";
import NodeTransformSchema from "@/generated/schemas/flowcraft_proto.v1.NodeTransformParams.schema.json";
import PromptGenSchema from "@/generated/schemas/flowcraft_proto.v1.PromptGenParams.schema.json";

/**
 * Registry for Protobuf message descriptors, used for serialization.
 * These are DescMessage objects, NOT JSON Schemas.
 */
export const SCHEMA_MAP: Record<string, DescMessage> = {
  [AddEdgeRequestSchema.typeName]: AddEdgeRequestSchema,
  [AddNodeRequestSchema.typeName]: AddNodeRequestSchema,
  [AddSubGraphRequestSchema.typeName]: AddSubGraphRequestSchema,
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
 * Map of Action IDs to their JSON Schemas for RJSF.
 */
export const ACTION_SCHEMAS: Record<string, RJSFSchema> = {
  "chat-action": ChatActionSchema as RJSFSchema,
  "image-enhance": ImageEnhanceSchema as RJSFSchema,
  "node-transform": NodeTransformSchema as RJSFSchema,
  "prompt-gen": PromptGenSchema as RJSFSchema,
};

/**
 * Retrieves a JSON Schema for RJSF form rendering.
 */
export function getTypedSchema(actionId: string): RJSFSchema | undefined {
  return ACTION_SCHEMAS[actionId];
}
