import { useCallback, useEffect, useRef } from "react";
import { useShallow } from "zustand/react/shallow";

import { useFlowStore } from "@/store/flowStore";
import { commit } from "@/store/orchestrator";
import { useNavigationStore } from "@/store/ui/navigationStore";
import { MutationSource } from "@/types";
import { GraphMapper } from "@/utils/graphMapper";
import { type BindingOptions, type SyncedLens } from "@/utils/lens-types";

/**
 * useSyncedBinding (V3.3 - Reactive Viewport Pull)
 */
export function useSyncedBinding<T>(lens: SyncedLens<T>, options: BindingOptions<T> = {}): [T, (newValue: T) => void] {
  const { spacetimeConn: conn } = useFlowStore();
  const activeScopeId = useNavigationStore((s) => s.activeScopeId);

  // 1. Controlled data pull (for non-resident listener data like Viewport)
  useEffect(() => {
    if (!conn || !lens.category) return;

    if (lens.category === "viewport") {
      const pullViewport = () => {
        const currentScope = activeScopeId || "root";
        let entry = null;
        for (const row of conn.db.viewportState.iter()) {
          if (row.id === currentScope) {
            entry = row;
            break;
          }
        }
        if (entry) {
          const remote = GraphMapper.toViewport(entry);
          if (remote) {
            useFlowStore.setState({ viewport: remote });
          }
        }
      };

      pullViewport();
    }
  }, [conn, lens.category, activeScopeId]);

  // 2. Zustand binding logic
  const value = useFlowStore(useShallow(lens.get));
  const lastKnownValueRef = useRef<T>(value);
  const isLocalUpdateRef = useRef(false);

  useEffect(() => {
    if (!isLocalUpdateRef.current && value !== lastKnownValueRef.current) {
      options.onIncoming?.(value, lastKnownValueRef.current);
    }
    lastKnownValueRef.current = value;
    isLocalUpdateRef.current = false;
  }, [value, options]);

  // 3. Outgoing logic
  const setValue = useCallback(
    (val: T) => {
      isLocalUpdateRef.current = true;
      lastKnownValueRef.current = val;
      commit(
        (draft) => {
          lens.set(draft as any, val);
        },
        {
          description: lens.description,
          isHistoryOp: options.undoable ?? lens.category !== "viewport",
          isInteractionEnd: true,
          source: MutationSource.SOURCE_USER,
          transient: options.transient,
        },
      );
    },
    [lens, options],
  );

  return [value, setValue];
}
