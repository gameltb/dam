import { useMemo } from "react";
import { useSyncedBinding } from "@/hooks/core/useSyncedBinding";

/**
 * useViewportProperty
 * 
 * Properties bound to the chart viewport (x, y, zoom).
 */
export function useViewportProperty<K extends "x" | "y" | "zoom">(property: K) {
  const lens = useMemo(() => ({
    category: 'viewport' as const,
    get: (s: any) => s.viewport[property],
    set: (d: any, val: number) => {
      d.viewport[property] = val;
    },
    description: `Update viewport ${property}`
  }), [property]);

  return useSyncedBinding(lens);
}