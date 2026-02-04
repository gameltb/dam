import { useEffect } from "react";
import { useFlowStore } from "@/store/flowStore";
import { useNavigationStore } from "@/store/ui/navigationStore";
import { type AppNode } from "@/types";
import { GraphMapper } from "@/utils/graphMapper";
import { 
  registerMaterializer, 
  getMaterializers 
} from "@/utils/materializerRegistry";

import { coreMaterializer } from "@/store/materializers/coreMaterializer";
import { nodeMaterializer } from "@/store/materializers/nodeMaterializer";
import { viewportMaterializer } from "@/store/materializers/viewportMaterializer";

// 1. Initialize registry
registerMaterializer(coreMaterializer);
registerMaterializer(nodeMaterializer);
registerMaterializer(viewportMaterializer);

/**
 * useGraphSync
 */
export const useGraphSync = (isActive: boolean) => {
  const { spacetimeConn: conn } = useFlowStore();
  const activeScopeId = useNavigationStore((s) => s.activeScopeId);

  useEffect(() => {
    if (!isActive || !conn) return;

    const reconcile = () => {
      const nodes: Record<string, AppNode> = { ...useFlowStore.getState().nodesById };
      for (const row of conn.db.nodes.iter()) {
        if (!nodes[row.nodeId]) nodes[row.nodeId] = GraphMapper.createSkeleton(row);
      }
      useFlowStore.setState({ nodesById: nodes });
    };

    const onInsert = (_ctx: any, row: any) => {
      useFlowStore.setState(s => ({
        nodesById: { ...s.nodesById, [row.nodeId]: GraphMapper.createSkeleton(row) }
      }));
    };

    const onDelete = (_ctx: any, row: any) => {
      useFlowStore.setState(s => {
        const next = { ...s.nodesById };
        delete next[row.nodeId];
        return { nodesById: next };
      });
    };

    conn.db.nodes.onInsert(onInsert);
    conn.db.nodes.onDelete(onDelete);

    reconcile();

    const cleanups = getMaterializers().map(m => m.setup(conn, activeScopeId));

    return () => {
      conn.db.nodes.removeOnInsert(onInsert);
      conn.db.nodes.removeOnDelete(onDelete);
      cleanups.forEach(cleanup => {
        if (typeof cleanup === 'function') cleanup();
      });
    };
  }, [isActive, conn, activeScopeId]);
};
