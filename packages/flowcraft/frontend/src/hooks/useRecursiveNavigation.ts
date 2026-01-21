import { useReactFlow, useStore } from "@xyflow/react";
import { useEffect, useRef } from "react";
import { useShallow } from "zustand/react/shallow";

import { useFlowStore } from "@/store/flowStore";
import { useUiStore } from "@/store/uiStore";
import { type DynamicNodeData, type GroupNodeData } from "@/types";

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

  const { activeScopeId, getViewportForScope, saveViewportForScope, setActiveScope } = useUiStore(
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
  const nodesRef = useRef(nodes);
  nodesRef.current = nodes;

  useEffect(() => {
    if (isTransitioning.current) return;

    const prevZoom = lastZoomRef.current;
    lastZoomRef.current = zoom;

    // 穿梭进入逻辑
    if (zoom > ENTER_ZOOM_THRESHOLD && prevZoom <= ENTER_ZOOM_THRESHOLD) {
      const flowCenter = screenToFlowPosition({ x: window.innerWidth / 2, y: window.innerHeight / 2 });

      const targetNode = nodesRef.current.find((n) => {
        const data = n.data as DynamicNodeData | GroupNodeData;
        const isLens = data.managedScopeId !== undefined || n.type === "groupNode";
        if (!isLens) return false;

        const w = n.measured?.width || 300;
        const h = n.measured?.height || 200;
        return (
          flowCenter.x > n.position.x &&
          flowCenter.x < n.position.x + w &&
          flowCenter.y > n.position.y &&
          flowCenter.y < n.position.y + h
        );
      });

      if (targetNode) {
        isTransitioning.current = true;

        // 1. 保存当前层级的视角
        saveViewportForScope(activeScopeId, { x, y, zoom });

        const data = targetNode.data as DynamicNodeData | GroupNodeData;
        const targetScope = data.managedScopeId || targetNode.id;
        setActiveScope(targetScope);

        // 2. 恢复目标层级的视角（如果有）
        const saved = getViewportForScope(targetScope);
        const nextViewport = saved || { x: window.innerWidth / 2, y: window.innerHeight / 2, zoom: 0.8 };

        setViewport(nextViewport, { duration: 500 }).finally(() => {
          isTransitioning.current = false;
        });
      }
    }

    // 穿梭退出逻辑
    if (zoom < EXIT_ZOOM_THRESHOLD && prevZoom >= EXIT_ZOOM_THRESHOLD && activeScopeId) {
      isTransitioning.current = true;

      // 1. 保存当前子层级的视角
      saveViewportForScope(activeScopeId, { x, y, zoom });

      // 2. 回退到顶层或父级
      // FIXME: Currently we only support root back-navigation. Need a parentId registry for deep nesting.
      const targetScope = null;
      setActiveScope(targetScope);

      // 3. 恢复父层级的视角
      const saved = getViewportForScope(targetScope);
      const nextViewport = saved || { x: window.innerWidth / 2, y: window.innerHeight / 2, zoom: 1.0 };

      setViewport(nextViewport, { duration: 500 }).finally(() => {
        isTransitioning.current = false;
      });
    }
  }, [
    zoom,
    x,
    y,
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
