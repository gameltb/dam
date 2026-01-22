import { type ReducerCtx, t } from "spacetimedb/server";

import { type NodeTemplate, NodeTemplateSchema } from "../generated/flowcraft/v1/core/node_pb";
import {
  InferenceConfigDiscoveryResponseSchema,
  type InferenceConfigDiscoveryResponse as ProtoInferenceConfigDiscoveryResponse,
} from "../generated/flowcraft/v1/core/service_pb";
import {
  services_InferenceConfigDiscoveryResponse as StdbInferenceConfigDiscoveryResponse,
  core_NodeTemplate as StdbNodeTemplate,
} from "../generated/generated_schema";
import { pbToStdb } from "../generated/proto-stdb-bridge";
import { type AppSchema } from "../schema";

export const configReducers = {
  register_template: {
    args: { template: NodeTemplateSchema },
    handler: (ctx: ReducerCtx<AppSchema>, { template }: { template: NodeTemplate }) => {
      const existing = ctx.db.nodeTemplates.templateId.find(template.templateId);
      const stdbState = pbToStdb(NodeTemplateSchema, StdbNodeTemplate, template) as StdbNodeTemplate;

      if (existing) {
        ctx.db.nodeTemplates.templateId.update({
          state: stdbState,
          templateId: template.templateId,
        });
      } else {
        ctx.db.nodeTemplates.insert({
          state: stdbState,
          templateId: template.templateId,
        });
      }
    },
  },

  update_inference_config: {
    args: {
      config: InferenceConfigDiscoveryResponseSchema,
      configId: t.string(),
    },
    handler: (
      ctx: ReducerCtx<AppSchema>,
      { config, configId }: { config: ProtoInferenceConfigDiscoveryResponse; configId: string },
    ) => {
      const existing = ctx.db.inferenceConfig.configId.find(configId);
      const stdbState = pbToStdb(
        InferenceConfigDiscoveryResponseSchema,
        StdbInferenceConfigDiscoveryResponse,
        config,
      ) as StdbInferenceConfigDiscoveryResponse;

      const record = {
        configId: configId,
        state: stdbState,
      };
      if (existing) {
        ctx.db.inferenceConfig.configId.update(record);
      } else {
        ctx.db.inferenceConfig.insert(record);
      }
    },
  },
};
