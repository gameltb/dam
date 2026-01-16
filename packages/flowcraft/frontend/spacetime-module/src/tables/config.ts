import { t, table } from "spacetimedb/server";

import { InferenceConfigDiscoveryResponse, NodeTemplate } from "../generated/generated_schema";

export const nodeTemplates = table(
  {
    name: "node_templates",
    public: true,
  },
  {
    state: NodeTemplate,
    templateId: t.string().primaryKey(),
  },
);

export const inferenceConfig = table(
  {
    name: "inference_config",
    public: true,
  },
  {
    configId: t.string().primaryKey(),
    state: InferenceConfigDiscoveryResponse,
  },
);
