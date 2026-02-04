import { useGraphSync } from "@/hooks/graph/useGraphSync";
import { useSpacetimeConnection } from "@/hooks/integration/useSpacetimeConnection";
import { useTaskSync } from "@/hooks/integration/useTaskSync";

/**
 * useSpacetimeSync
 *
 * Coordinator hook for SpacetimeDB synchronization.
 * Delegates specific syncing logic to specialized sub-hooks.
 */
export const useSpacetimeSync = () => {
  const { isActive } = useSpacetimeConnection();

  // Task & Chat Sync
  useTaskSync(isActive);

  // Graph & Viewport Sync
  useGraphSync(isActive);
};
