/**
 * reconcileMap (V2.0)
 *
 * Optimized for precision and performance.
 * Avoids JSON.stringify by using custom equality checks.
 */
export function reconcileMap<R, D>(
  remoteItems: readonly R[],
  localMap: Record<string, D>,
  getId: (item: R) => string,
  mapper: (item: R, existing?: D) => D | null,
  options: {
    hasChanged: (existing: D, remote: R) => boolean; // Precise change detection
    onChanged?: () => void;
    shouldSkip?: (existing: D) => boolean;
  },
): Record<string, D> {
  const nextMap = { ...localMap };
  const remoteIds = new Set<string>();
  let hasChanges = false;

  remoteItems.forEach((remoteItem) => {
    const id = getId(remoteItem);
    remoteIds.add(id);

    const existing = nextMap[id];

    if (existing) {
      if (options.shouldSkip?.(existing)) return;

      // PRECISION: Only map and update if the remote row actually differs
      if (options.hasChanged(existing, remoteItem)) {
        const domainObj = mapper(remoteItem, existing);
        if (domainObj) {
          nextMap[id] = domainObj;
          hasChanges = true;
        }
      }
    } else {
      // ADD
      const domainObj = mapper(remoteItem);
      if (domainObj) {
        nextMap[id] = domainObj;
        hasChanges = true;
      }
    }
  });

  // REMOVE
  Object.keys(nextMap).forEach((id) => {
    if (!remoteIds.has(id)) {
      delete nextMap[id];
      hasChanges = true;
    }
  });

  if (hasChanges) options.onChanged?.();
  return nextMap;
}
