import { useEffect } from "react";
import { type Infer } from "spacetimedb";

import { type NodesRow } from "@/generated/spacetime";
import { useFlowStore } from "@/store/flowStore";
import { coreMaterializer } from "@/store/materializers/coreMaterializer";
import { nodeMaterializer } from "@/store/materializers/nodeMaterializer";
import { viewportMaterializer } from "@/store/materializers/viewportMaterializer";
import { useNavigationStore } from "@/store/ui/navigationStore";
import { type AppNode } from "@/types";
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

    const reconcile = () => {
      const nodes: Record<string, AppNode> = { ...useFlowStore.getState().nodesById };
      for (const row of conn.db.nodes.iter()) {
        nodes[row.nodeId] ??= GraphMapper.createSkeleton(row);
      }
      useFlowStore.setState({ nodesById: nodes });
    };

    const onInsert = (_ctx: unknown, row: Infer<typeof NodesRow>) => {
      applySyncInsert(row.nodeId, GraphMapper.createSkeleton(row));
    };

    const onDelete = (_ctx: unknown, row: Infer<typeof NodesRow>) => {
      applySyncDelete(row.nodeId);
    };

    conn.db.nodes.onInsert(onInsert);
    conn.db.nodes.onDelete(onDelete);

    reconcile();

    const cleanups = getMaterializers().map((m) => m.setup(conn, activeScopeId));

    return () => {
      conn.db.nodes.removeOnInsert(onInsert);
      conn.db.nodes.removeOnDelete(onDelete);
      cleanups.forEach((cleanup) => {
        if (typeof cleanup === "function") cleanup();
      });
    };
  }, [isActive, conn, activeScopeId]);
};
