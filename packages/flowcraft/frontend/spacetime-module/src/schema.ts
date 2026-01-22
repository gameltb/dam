import { type InferSchema, schema } from "spacetimedb/server";

import { edges, nodes, viewportState } from "./tables/base";
import { chatMessages, chatStreams } from "./tables/chat";
import { inferenceConfig, nodeTemplates } from "./tables/config";
import { nodeRuntimeStates } from "./tables/runtime";
import {
  clientTaskAssignments,
  nodeSignals,
  operationLogs,
  taskAuditLog,
  tasks,
  widgetValues,
  workers,
} from "./tables/task";

export const spacetimedb = schema(
  nodes,
  edges,
  viewportState,
  chatMessages,
  chatStreams,
  tasks,
  workers,
  taskAuditLog,
  nodeSignals,
  widgetValues,
  clientTaskAssignments,
  nodeTemplates,
  inferenceConfig,
  operationLogs,
  nodeRuntimeStates,
);

export type AppSchema = InferSchema<typeof spacetimedb>;
