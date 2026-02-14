import { useReactFlow, useStore } from "@xyflow/react";
import { useEffect, useRef } from "react";
import { useShallow } from "zustand/react/shallow";

import { useFlowStore } from "@/store/flowStore";
import { useNavigationStore } from "@/store/ui/navigationStore";
import { AppNodeType, type DynamicNodeData, type GroupNodeData } from "@/types";

const ENTER_ZOOM_THRESHOLD = 2.0;
const EXIT_ZOOM_THRESHOLD = 0.3;

export const useRecursiveNavigation = () => {
  const { x, y, zoom } = useStore(
    useShallow((s) => ({
      x: s.transform[0],
      y: s.transform[1],
      zoom: s.transform[2],
    })),
  );

  const { activeScopeId, getViewportForScope, saveViewportForScope, setActiveScope } = useNavigationStore(
    useShallow((s) => ({
      activeScopeId: s.activeScopeId,
      getViewportForScope: s.getViewportForScope,
      saveViewportForScope: s.saveViewportForScope,
      setActiveScope: s.setActiveScope,
    })),
  );
  const { nodes, refreshView } = useFlowStore(
    useShallow((s) => ({
      nodes: s.nodes,
      refreshView: s.refreshView,
    })),
  );
  const { screenToFlowPosition, setViewport } = useReactFlow();

  const lastZoomRef = useRef(zoom);
  const isTransitioning = useRef(false);

  useEffect(() => {
    if (isTransitioning.current) return;

    const prevZoom = lastZoomRef.current;
    lastZoomRef.current = zoom;

    if (zoom > ENTER_ZOOM_THRESHOLD && prevZoom <= ENTER_ZOOM_THRESHOLD) {
      const flowCenter = screenToFlowPosition({ x: window.innerWidth / 2, y: window.innerHeight / 2 });

      const targetNode = nodes.find((n) => {
        const data = n.data as DynamicNodeData | GroupNodeData;
        const isLens = data.managedScopeId !== undefined || n.type === AppNodeType.GROUP;
        if (!isLens) return false;

        const w = n.measured?.width ?? 300;
        const h = n.measured?.height ?? 200;
        return (
          flowCenter.x > n.position.x &&
          flowCenter.x < n.position.x + w &&
          flowCenter.y > n.position.y &&
          flowCenter.y < n.position.y + h
        );
      });

      if (targetNode) {
        isTransitioning.current = true;

        saveViewportForScope(activeScopeId, { x, y, zoom });

        const data = targetNode.data as DynamicNodeData | GroupNodeData;
        const targetScope = data.managedScopeId ?? targetNode.id;
        setActiveScope(targetScope);

        const saved = getViewportForScope(targetScope);
        const nextViewport = saved ?? { x: window.innerWidth / 2, y: window.innerHeight / 2, zoom: 0.8 };

        void setViewport(nextViewport, { duration: 500 }).finally(() => {
          isTransitioning.current = false;
        });
      }
    }

    if (zoom < EXIT_ZOOM_THRESHOLD && prevZoom >= EXIT_ZOOM_THRESHOLD && activeScopeId) {
      isTransitioning.current = true;

      saveViewportForScope(activeScopeId, { x, y, zoom });

      const targetScope = null;
      setActiveScope(targetScope);

      const saved = getViewportForScope(targetScope);
      const nextViewport = saved ?? { x: window.innerWidth / 2, y: window.innerHeight / 2, zoom: 1.0 };

      void setViewport(nextViewport, { duration: 500 }).finally(() => {
        isTransitioning.current = false;
      });
    }
  }, [
    zoom,
    x,
    y,
    nodes,
    activeScopeId,
    setActiveScope,
    setViewport,
    screenToFlowPosition,
    saveViewportForScope,
    getViewportForScope,
  ]);

  const lastScopeIdRef = useRef<null | string | undefined>(undefined);
  useEffect(() => {
    if (activeScopeId !== lastScopeIdRef.current) {
      refreshView();
      lastScopeIdRef.current = activeScopeId;
    }
  }, [activeScopeId, refreshView]);
};
