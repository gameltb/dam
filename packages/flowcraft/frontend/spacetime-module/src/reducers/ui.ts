import { type ReducerCtx, t } from "spacetimedb/server";

import { type AppSchema } from "../schema";

/**
 * UI-specific reducers for high-frequency transient state synchronization.
 */
export const uiReducers = {
  update_node_ui_state: {
    args: { nodeId: t.string(), stateJson: t.string() },
    handler: (ctx: ReducerCtx<AppSchema>, { nodeId, stateJson }: { nodeId: string; stateJson: string }) => {
      const existing = ctx.db.nodeUiState.nodeId.find(nodeId);

      if (existing) {
        ctx.db.nodeUiState.nodeId.update({
          lastUpdated: BigInt(Date.now()),
          nodeId,
          stateJson,
        });
      } else {
        ctx.db.nodeUiState.insert({
          lastUpdated: BigInt(Date.now()),
          nodeId,
          stateJson,
        });
      }
    },
  },
};
