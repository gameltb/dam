import { onSpacetimeConnect } from "@/server/spacetimeClient";

import { inferenceService } from "./InferenceService";
import { NodeRegistry } from "./NodeRegistry";

export const initConfigSync = () => {
  onSpacetimeConnect((conn) => {
    console.log("[ConfigSync] Syncing templates and inference config...");

    // 1. Sync Templates
    const templates = NodeRegistry.getTemplates();
    templates.forEach((tmpl) => {
      conn.pbreducers.registerTemplate({
        template: tmpl,
      });
    });

    // 2. Sync Inference Config
    const config = inferenceService.getConfig();
    conn.pbreducers.updateInferenceConfig({
      config: config,
      configId: "default",
    });
  });
};
