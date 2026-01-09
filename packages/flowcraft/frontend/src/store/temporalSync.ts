import { type TemporalState } from "zundo";
import { type StoreApi, type UseBoundStore } from "zustand";
import { useStoreWithEqualityFn } from "zustand/traditional";

import { type RFState } from "@/store/types";
import { dehydrateNode } from "@/utils/nodeUtils";

import { ydoc, yEdges, yNodes } from "./yjsInstance";

let isSyncingFromTemporal = false;
let lastPastLength = 0;
let lastFutureLength = 0;

export function setupTemporalSync(
  useStore: UseBoundStore<StoreApi<RFState>> & {
    temporal: StoreApi<TemporalState<RFState>>;
  },
) {
  useStore.temporal.subscribe((state: TemporalState<RFState>) => {
    // Only sync back to Yjs if an actual undo or redo happened.
    const isUndo = state.pastStates.length < lastPastLength;
    const isRedo = state.futureStates.length < lastFutureLength;

    lastPastLength = state.pastStates.length;
    lastFutureLength = state.futureStates.length;

    if (isUndo || isRedo) {
      if (isSyncingFromTemporal) return;
      isSyncingFromTemporal = true;
      try {
        const current = useStore.getState();
        useStore.setState({ isLayoutDirty: true });
        ydoc.transact(() => {
          yNodes.clear();
          yEdges.clear();
          // We dehydrate nodes before putting them back into Yjs
          current.nodes.forEach((n) => yNodes.set(n.id, dehydrateNode(n)));
          current.edges.forEach((e) => yEdges.set(e.id, e));
        }, "undo-redo");
      } finally {
        isSyncingFromTemporal = false;
      }
    }
  });
}

export function useTemporalStore<T>(
  useStore: UseBoundStore<StoreApi<RFState>> & {
    temporal: StoreApi<TemporalState<RFState>>;
  },
  selector: (state: TemporalState<RFState>) => T,
  equality?: (a: T, b: T) => boolean,
): T {
  return useStoreWithEqualityFn(
    useStore.temporal,
    (state) => selector(state),
    equality,
  );
}
