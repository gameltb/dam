import { type InferSchema, schema } from "spacetimedb/server";

import { edges, nodes, viewportState } from "./tables/base";
import { chatMessages, chatStreams } from "./tables/chat";
import { inferenceConfig, nodeTemplates } from "./tables/config";
import { clientTaskAssignments, nodeSignals, operationLogs, tasks, widgetValues } from "./tables/task";

export const spacetimedb = schema(
  nodes,
  edges,
  viewportState,
  chatMessages,
  chatStreams,
  tasks,
  nodeSignals,
  widgetValues,
  clientTaskAssignments,
  nodeTemplates,
  inferenceConfig,
  operationLogs,
);

export type AppSchema = InferSchema<typeof spacetimedb>;
