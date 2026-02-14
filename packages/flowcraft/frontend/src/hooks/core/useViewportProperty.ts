import { useMemo } from "react";

import { useSyncedBinding } from "@/hooks/core/useSyncedBinding";

/**
 * useViewportProperty
 *
 * Properties bound to the chart viewport (x, y, zoom).
 */
export function useViewportProperty<K extends "x" | "y" | "zoom">(property: K) {
  const lens = useMemo(
    () => ({
      category: "viewport" as const,
      description: `Update viewport ${property}`,
      get: (s: any) => s.viewport[property],
      set: (d: any, val: number) => {
        d.viewport[property] = val;
      },
    }),
    [property],
  );

  return useSyncedBinding(lens);
}
