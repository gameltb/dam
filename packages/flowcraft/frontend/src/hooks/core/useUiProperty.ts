import { useMemo } from "react";

import { type SyncedLens } from "@/utils/lens-types";

import { useSyncedBinding } from "@/hooks/core/useSyncedBinding";

/**
 * useUiProperty
 *
 * It is now just a special case of useSyncedBinding.
 * We specify a different synchronization backend via the Lens category.
 */
export function useUiProperty<T>(nodeId: string, key: string, defaultValue: T) {
  const lens = useMemo(
    (): SyncedLens<T> => ({
      category: "ui",
      description: `UI state update for ${nodeId}: ${key}`,
      get: (_s: any) => defaultValue, // This get will not be used in table mode
      set: () => {},
    }),
    [nodeId, key, defaultValue],
  );

  return useSyncedBinding(lens, { backend: "table" });
}
