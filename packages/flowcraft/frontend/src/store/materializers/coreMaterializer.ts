import { useFlowStore } from "@/store/flowStore";
import { log } from "@/utils/logger";
import { type PbConnection } from "@/utils/pb-client";

/**
 * CoreMaterializer
 * Handles global/static metadata that is not scoped to a specific subgraph.
 */
export const coreMaterializer = {
  name: "core",

  setup: (conn: PbConnection) => {
    log.debug("Sync", "Core Materializer Setup");

    // 1. Subscribe to global metadata
    const sub = conn
      .subscriptionBuilder()
      .onApplied(() => {
        log.debug("Sync", "Global Metadata Applied");
        // Force refresh view to ensure templates are propagated to menus
        useFlowStore.getState().refreshView();
      })
      .onError((ctx) => {
        log.error("Sync", `Core Subscription Error: ${ctx.event}`);
      })
      .subscribe([
        "SELECT * FROM node_templates",
        "SELECT * FROM inference_config",
        "SELECT * FROM workers",
        "SELECT * FROM tasks",
        "SELECT * FROM chat_messages",
        "SELECT * FROM chat_streams",
      ]);

    return () => {
      if (process.env.NODE_ENV === "development") {
        console.debug("[Sync] Core Materializer Cleanup");
      }
      sub.unsubscribe();
    };
  },
};
