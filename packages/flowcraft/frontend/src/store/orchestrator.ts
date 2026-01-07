import { pipeline } from "./middleware/pipeline";
import { syncMiddleware } from "./middleware/syncMiddleware";
import { taskMiddleware } from "./middleware/taskMiddleware";

/**
 * Orchestrator
 *
 * Listens to store events and handles cross-store side effects.
 * Most graph modification logic has been moved to unified middlewares in applyMutations.
 */
export function initStoreOrchestrator() {
  // Initialize Mutation Pipeline
  pipeline.clear().use(taskMiddleware).use(syncMiddleware);

  // Currently, most cross-store sync is handled via Middlewares in flowStore.ts
  // You can add non-mutation related listeners here (e.g., UI specific triggers)
}
