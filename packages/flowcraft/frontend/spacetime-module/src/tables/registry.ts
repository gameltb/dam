import { t, table } from "spacetimedb/server";

export const nodeTemplates = table(
  {
    name: "node_templates",
    public: true,
  },
  {
    state: Object.assign(t.byteArray(), { __pb_schema: "NodeTemplate" }),
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
    state: Object.assign(t.byteArray(), { __pb_schema: "InferenceConfigDiscoveryResponse" }),
  },
);
