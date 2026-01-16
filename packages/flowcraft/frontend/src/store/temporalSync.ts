import { type TemporalState } from "zundo";
import { type StoreApi, type UseBoundStore } from "zustand";
import { useStoreWithEqualityFn } from "zustand/traditional";

import { type RFState } from "@/store/types";

export function useTemporalStore<T>(
  useStore: UseBoundStore<StoreApi<RFState>> & {
    temporal: StoreApi<TemporalState<RFState>>;
  },
  selector: (state: TemporalState<RFState>) => T,
  equality?: (a: T, b: T) => boolean,
): T {
  return useStoreWithEqualityFn(useStore.temporal, (state) => selector(state), equality);
}
