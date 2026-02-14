import { useFlowStore } from "@/store/flowStore";
import { type NavigationState, useNavigationStore } from "@/store/ui/navigationStore";
import { Scope } from "@/types";
import { GraphMapper } from "@/utils/graphMapper";
import { type SyncedLens } from "@/utils/lens-types";
import { log } from "@/utils/logger";
import { type PbConnection } from "@/utils/pb-client";

/**
 * ViewportMaterializer
 */
export const viewportMaterializer = {
  name: "viewport",
  setup: (conn: PbConnection) => {
    // Listen for changes in global activeScopeId to perform data fetching
    const unsubscribe = useNavigationStore.subscribe((state: NavigationState) => {
      const currentScope = state.activeScopeId || Scope.ROOT;
      for (const row of conn.db.viewportState.iter()) {
        if (row.id === currentScope) {
          const remote = GraphMapper.toViewport(row);
          if (remote) {
            log.debug("Sync", "Viewport Update", remote);
            useFlowStore.setState({ viewport: remote });
          }
          break;
        }
      }
    });
    return () => {
      unsubscribe();
    };
  },

  tables: ["viewport_state"],
};

export const ViewportLenses = {
  prop: <K extends "x" | "y" | "zoom">(property: K): SyncedLens<number> => ({
    category: "viewport",
    description: `Update viewport ${property}`,
    get: (s) => s.viewport[property],
    set: (d, val) => {
      d.viewport[property] = val;
    },
  }),
};
