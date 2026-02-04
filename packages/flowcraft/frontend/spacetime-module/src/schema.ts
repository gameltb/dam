import { type InferSchema, schema } from "spacetimedb/server";

import { nodeSignals, operationLogs, taskAuditLog } from "./tables/audit";
import { chatMessages, chatStreams } from "./tables/chat";
import {
  edges,
  nodeData,
  nodeMetadata,
  nodes,
  nodeTransforms,
  nodeUiState,
  viewportState,
  widgetValues,
} from "./tables/core";
import { clientTaskAssignments, nodeRuntimeStates, tasks, workers } from "./tables/kernel";
import { inferenceConfig, nodeTemplates } from "./tables/registry";

export const spacetimedb = schema(
  nodes,
  nodeTransforms,
  nodeMetadata,
  nodeData,
  edges,
  viewportState,
  chatMessages,
  chatStreams,
  tasks,
  workers,
  taskAuditLog,
  nodeSignals,
  widgetValues,
  nodeUiState,
  clientTaskAssignments,
  nodeTemplates,
  inferenceConfig,
  operationLogs,
  nodeRuntimeStates,
);

export type AppSchema = InferSchema<typeof spacetimedb>;
