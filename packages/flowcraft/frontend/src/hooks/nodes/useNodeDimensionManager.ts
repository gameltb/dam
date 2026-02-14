import { useEffect, useRef } from "react";

import { editNode } from "@/store/orchestrator";

export enum SizingStrategy {
  ASPECT_RATIO = "aspect-ratio",
  CONTENT_FIT = "content-fit",
  MANUAL = "manual",
}

interface SizingOptions {
  ratio?: number;
  strategy?: SizingStrategy;
}

/**
 * useNodeDimensionManager
 *
 * Core business logic: Monitor DOM dimension changes and sync to state machine according to strategy.
 */
export function useNodeDimensionManager(
  nodeId: string,
  contentRef: React.RefObject<HTMLElement | null>,
  options: SizingOptions,
) {
  const { ratio, strategy = SizingStrategy.MANUAL } = options;
  const lastUpdate = useRef<{ h: number; w: number }>({ h: 0, w: 0 });

  useEffect(() => {
    if (!contentRef.current || strategy === SizingStrategy.MANUAL) return;

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) return;

      const { height, width } = entry.contentRect;
      const targetWidth = width;
      let targetHeight = height;

      if (strategy === SizingStrategy.ASPECT_RATIO && ratio) {
        targetHeight = width / ratio;
      }

      // Debounce: Only commit when dimension changes exceed threshold
      if (Math.abs(lastUpdate.current.w - targetWidth) > 2 || Math.abs(lastUpdate.current.h - targetHeight) > 2) {
        lastUpdate.current = { h: targetHeight, w: targetWidth };

        editNode(nodeId, (draft) => {
          draft.width = targetWidth;
          draft.height = targetHeight;
          if (draft.presentation) {
            draft.presentation.width = targetWidth;
            draft.presentation.height = targetHeight;
          }
        });
      }
    });

    observer.observe(contentRef.current);
    return () => {
      observer.disconnect();
    };
  }, [nodeId, strategy, ratio, contentRef]);
}
