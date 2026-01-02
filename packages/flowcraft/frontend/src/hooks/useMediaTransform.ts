import { useState, useCallback, useEffect } from "react";

export const useMediaTransform = (activeIndex: number) => {
  const [zoom, setZoom] = useState(1);
  const [rotation, setRotate] = useState(0);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  const resetTransform = useCallback(() => {
    setZoom(1);
    setRotate(0);
    setOffset({ x: 0, y: 0 });
  }, []);

  useEffect(() => {
    const raf = requestAnimationFrame(() => {
      resetTransform();
    });
    return () => {
      cancelAnimationFrame(raf);
    };
  }, [activeIndex, resetTransform]);

  const handleZoomIn = (e?: React.MouseEvent) => {
    e?.stopPropagation();
    setZoom((prev) => Math.min(prev + 0.25, 5));
  };

  const handleZoomOut = (e?: React.MouseEvent) => {
    e?.stopPropagation();
    setZoom((prev) => Math.max(prev - 0.25, 0.5));
  };

  const handleRotate = (e?: React.MouseEvent) => {
    e?.stopPropagation();
    setRotate((prev) => prev + 90);
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (zoom <= 1) return;
    e.preventDefault();
    setIsDragging(true);
    setDragStart({ x: e.clientX - offset.x, y: e.clientY - offset.y });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;
    setOffset({
      x: e.clientX - dragStart.x,
      y: e.clientY - dragStart.y,
    });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleWheel = (e: React.WheelEvent) => {
    if (e.deltaY < 0) handleZoomIn();
    else handleZoomOut();
  };

  return {
    zoom,
    rotation,
    offset,
    isDragging,
    resetTransform,
    handleZoomIn,
    handleZoomOut,
    handleRotate,
    handleMouseDown,
    handleMouseMove,
    handleMouseUp,
    handleWheel,
  };
};
