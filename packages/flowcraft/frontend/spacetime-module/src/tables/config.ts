import { t, table } from "spacetimedb/server";

import { services_InferenceConfigDiscoveryResponse, core_NodeTemplate } from "../generated/generated_schema";

export const nodeTemplates = table(
  {
    name: "node_templates",
    public: true,
  },
  {
    state: core_NodeTemplate,
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
    state: services_InferenceConfigDiscoveryResponse,
  },
);