import { useEffect } from "react";
import { type Infer } from "spacetimedb";

import { type NodesRow } from "@/generated/spacetime";
import { useFlowStore } from "@/store/flowStore";
import { coreMaterializer } from "@/store/materializers/coreMaterializer";
import { nodeMaterializer } from "@/store/materializers/nodeMaterializer";
import { viewportMaterializer } from "@/store/materializers/viewportMaterializer";
import { useNavigationStore } from "@/store/ui/navigationStore";
import { GraphMapper } from "@/utils/graphMapper";
import { getMaterializers, registerMaterializer } from "@/utils/materializerRegistry";
import { applySyncDelete, applySyncInsert } from "@/utils/syncUtils";

registerMaterializer(coreMaterializer);
registerMaterializer(nodeMaterializer);
registerMaterializer(viewportMaterializer);

export const useGraphSync = (isActive: boolean) => {
  const { spacetimeConn: conn } = useFlowStore();
  const activeScopeId = useNavigationStore((s) => s.activeScopeId);

  useEffect(() => {
    if (!isActive || !conn) return;

    const onInsert = (_ctx: unknown, row: Infer<typeof NodesRow>) => {
      applySyncInsert(row.nodeId, GraphMapper.createSkeleton(row));
    };

    const onDelete = (_ctx: unknown, row: Infer<typeof NodesRow>) => {
      applySyncDelete(row.nodeId);
    };

    conn.db.nodes.onInsert(onInsert);
    conn.db.nodes.onDelete(onDelete);

    // Initial pass (might be partial)
    useFlowStore.getState().syncWithDatabase();

    // The materializers handles the subscription and the final onApplied call
    const cleanups = getMaterializers().map((m) => {
      const materializer = m.setup(conn, activeScopeId);
      // If the materializer provided a specialized reconcile, we might use it,
      // but here we use our unified deepReconcile on subscription applied.
      return materializer;
    });

    // Re-run deep reconcile when subscription is fully ready
    // We listen to the subscription applied event via a custom listener or
    // simply rely on the materializers to trigger it.
    // To be truly reliable, we'll attach a one-time handler if the SDK supports it,
    // or let the materializer's onApplied handle the refresh.

    return () => {
      conn.db.nodes.removeOnInsert(onInsert);
      conn.db.nodes.removeOnDelete(onDelete);
      cleanups.forEach((cleanup) => {
        if (typeof cleanup === "function") cleanup();
      });
    };
  }, [isActive, conn, activeScopeId]);
};
