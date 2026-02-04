/** AUTO-GENERATED - DO NOT EDIT **/ 
/* eslint-disable */
import { type GenMessage } from "@bufbuild/protobuf/codegenv2";
import { ChatSyncMessageSchema } from "@/generated/flowcraft/v1/actions/chat_actions_pb";
import { EdgeSchema, NodeDataSchema, NodeSchema, NodeTemplateSchema } from "@/generated/flowcraft/v1/core/node_pb";
import { AddSubGraphRequestSchema, ChatMessageSchema, InferenceConfigDiscoveryResponseSchema, ResetNodeRequestSchema } from "@/generated/flowcraft/v1/core/service_pb";
import { TaskAuditLogSchema, TaskDefinitionSchema, TaskUpdateSchema, WorkerInfoSchema } from "@/generated/flowcraft/v1/core/kernel_pb";
import { ViewportSchema } from "@/generated/flowcraft/v1/core/base_pb";
import { ActionExecutionRequestSchema } from "@/generated/flowcraft/v1/core/action_pb";
import { NodeSignalSchema } from "@/generated/flowcraft/v1/core/signals_pb";

import { type DbConnection } from "./spacetime";

/**
 * PB Override Manifest
 */
export const PB_REDUCERS_MAP = {
  "addChatMessage": {
      message: { schema: ChatSyncMessageSchema } 
  },
  "registerTemplate": {
      template: { schema: NodeTemplateSchema } 
  },
  "updateInferenceConfig": {
      config: { schema: InferenceConfigDiscoveryResponseSchema } 
  },
  "logTaskEvent": {
      log: { schema: TaskAuditLogSchema } 
  },
  "registerWorker": {
      info: { schema: WorkerInfoSchema } 
  },
  "submitTask": {
      task: { schema: TaskDefinitionSchema } 
  },
  "updateTaskProgress": {
      update: { schema: TaskUpdateSchema } 
  },
  "addEdgePb": {
      edge: { schema: EdgeSchema } 
  },
  "addSubGraphPb": {
      req: { schema: AddSubGraphRequestSchema } 
  },
  "createNodePb": {
      node: { schema: NodeSchema } 
  },
  "setNodeDataPb": {
      state: { schema: NodeDataSchema } 
  },
  "updateViewport": {
      viewport: { schema: ViewportSchema } 
  },
  "resetNode": {
      req: { schema: ResetNodeRequestSchema } 
  },
  "executeAction": {
      request: { schema: ActionExecutionRequestSchema } 
  },
  "sendNodeSignal": {
      signal: { schema: NodeSignalSchema } 
  },
  "updateTaskStatus": {
      update: { schema: TaskUpdateSchema } 
  } 
} as const;

/**
 * Mapping between Tables and Protobuf Schemas
 */
export const TABLE_TO_PROTO: Record<string, { schema: GenMessage<any>, field: string }> = {
  "nodeSignals": { schema: NodeSignalSchema, field: "payload" },
  "chatMessages": { schema: ChatMessageSchema, field: "state" },
  "edges": { schema: EdgeSchema, field: "state" },
  "nodeData": { schema: NodeDataSchema, field: "state" },
  "viewportState": { schema: ViewportSchema, field: "state" },
  "nodeTemplates": { schema: NodeTemplateSchema, field: "state" },
  "inferenceConfig": { schema: InferenceConfigDiscoveryResponseSchema, field: "state" } 
} as const;

/**
 * Compile-time type safety assertion: ensures all mapped Reducers exist in the SDK
 */
type AssertReducersExist = keyof typeof PB_REDUCERS_MAP extends keyof DbConnection["reducers"]
  ? true
  : never;
export const _ASSERT_REDUCERS_SAFE: AssertReducersExist = true;
