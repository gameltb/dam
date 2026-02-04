import { useEffect, useRef, useState } from "react";

/**
 * useNodeVisibility
 * Monitors whether a node has ever entered the viewport.
 * Used to implement lazy-loading for node rendering.
 */
export const useNodeVisibility = (elementRef: React.RefObject<HTMLElement | null>) => {
  const [hasBeenVisible, setHasBeenVisible] = useState(false);
  const observerRef = useRef<IntersectionObserver | null>(null);

  useEffect(() => {
    if (hasBeenVisible || !elementRef.current) return;

    observerRef.current = new IntersectionObserver(
      ([entry]) => {
        if (entry?.isIntersecting) {
          setHasBeenVisible(true);
          // Once visible, disconnect the observer to achieve 'one-time activation'
          observerRef.current?.disconnect();
        }
      },
      {
        // Reserve a buffer (e.g., 200px) so users don't perceive the loading process
        rootMargin: "200px",
        threshold: 0.01,
      },
    );

    observerRef.current.observe(elementRef.current);

    return () => {
      observerRef.current?.disconnect();
    };
  }, [elementRef, hasBeenVisible]);

  return hasBeenVisible;
};
