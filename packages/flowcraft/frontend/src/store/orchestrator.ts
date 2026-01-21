import { pipeline } from "./middleware/pipeline";
import { taskMiddleware } from "./middleware/taskMiddleware";

/**
 * Orchestrator
 *
 * Listens to store events and handles cross-store side effects.
 * Most graph modification logic has been moved to unified middlewares in applyMutations.
 */
export function initStoreOrchestrator() {
  // Initialize Mutation Pipeline
  pipeline.clear().use(taskMiddleware);

  return pipeline;
}
