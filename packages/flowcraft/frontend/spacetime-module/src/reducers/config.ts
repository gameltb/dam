import { type ReducerCtx, t } from "spacetimedb/server";

import { type NodeTemplate, NodeTemplateSchema } from "../generated/flowcraft/v1/core/node_pb";
import { InferenceConfigDiscoveryResponseSchema } from "../generated/flowcraft/v1/core/service_pb";
import { type AppSchema } from "../schema";

const ConfigArg = InferenceConfigDiscoveryResponseSchema;

export const configReducers = {
  register_template: {
    args: { template: NodeTemplateSchema },
    handler: (
      ctx: ReducerCtx<AppSchema>,
      { template, templateBinary }: { template: NodeTemplate; templateBinary: Uint8Array },
    ) => {
      const existing = ctx.db.nodeTemplates.templateId.find(template.templateId);

      if (existing) {
        ctx.db.nodeTemplates.templateId.update({
          state: templateBinary,
          templateId: template.templateId,
        });
      } else {
        ctx.db.nodeTemplates.insert({
          state: templateBinary,
          templateId: template.templateId,
        });
      }
    },
  },

  update_inference_config: {
    args: { config: ConfigArg, configId: t.string() },
    handler: (
      ctx: ReducerCtx<AppSchema>,
      { configBinary, configId }: { configBinary: Uint8Array; configId: string },
    ) => {
      const existing = ctx.db.inferenceConfig.configId.find(configId);

      const record = {
        configId: configId,
        state: configBinary,
      };
      if (existing) {
        ctx.db.inferenceConfig.configId.update(record);
      } else {
        ctx.db.inferenceConfig.insert(record);
      }
    },
  },
};
