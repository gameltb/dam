import React, { useEffect, useState, useCallback, useMemo } from "react";
import { flowcraft_proto } from "../../generated/flowcraft_proto";
import type { AppNode } from "../../types";

const MediaType = flowcraft_proto.v1.MediaType;

interface MediaPreviewProps {
  node: AppNode;
  initialIndex: number;
  onClose: () => void;
}

export const MediaPreview: React.FC<MediaPreviewProps> = ({
  node,
  initialIndex,
  onClose,
}) => {
  const [activeIndex, setActiveIndex] = useState(initialIndex);
  const [isLoading, setIsLoading] = useState(false); // Default to false, will be set true only if needed

  const media = node.type === "dynamic" ? node.data.media : null;
  const items = useMemo(() => {
    if (!media) return [];
    return [media.url, ...(media.galleryUrls ?? [])].filter(
      Boolean,
    ) as string[];
  }, [media]);

  const currentUrl = items[activeIndex];

  // Preloading Logic
  useEffect(() => {
    const preload = (url: string) => {
      if (!url) return;
      const img = new Image();
      img.src = url;
    };

    // Preload next and prev
    if (activeIndex < items.length - 1) preload(items[activeIndex + 1]);
    if (activeIndex > 0) preload(items[activeIndex - 1]);
  }, [activeIndex, items]);

  const handleSwitch = useCallback(
    (newIndex: number) => {
      const url = items[newIndex];
      // For images, we can check if they are already in cache
      if (media?.type === MediaType.MEDIA_IMAGE) {
        const img = new Image();
        img.src = url;
        if (!img.complete) {
          setIsLoading(true);
        } else {
          setIsLoading(false);
        }
      } else {
        // Videos usually benefit from a loading indicator as buffering always takes a bit
        setIsLoading(true);
      }
      setActiveIndex(newIndex);
    },
    [items, media?.type],
  );

  const handlePrev = useCallback(() => {
    if (activeIndex > 0) handleSwitch(activeIndex - 1);
  }, [activeIndex, handleSwitch]);

  const handleNext = useCallback(() => {
    if (activeIndex < items.length - 1) handleSwitch(activeIndex + 1);
  }, [activeIndex, items.length, handleSwitch]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      if (e.key === "ArrowLeft") handlePrev();
      if (e.key === "ArrowRight") handleNext();
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [handlePrev, handleNext, onClose]);

  if (!media || items.length === 0) return null;

  return (
    <div
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        width: "100vw",
        height: "100vh",
        backgroundColor: "rgba(0,0,0,0.95)",
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        alignItems: "center",
        zIndex: 5000,
        backdropFilter: "blur(25px)",
      }}
    >
      {/* Header */}
      <div
        style={{
          position: "absolute",
          top: 0,
          width: "100%",
          padding: "25px 40px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          color: "white",
          background:
            "linear-gradient(to bottom, rgba(0,0,0,0.7), transparent)",
          boxSizing: "border-box",
          zIndex: 10,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "15px" }}>
          <span
            style={{
              fontSize: "20px",
              fontWeight: 600,
              letterSpacing: "-0.5px",
            }}
          >
            {node.data.label}
          </span>
          <div
            style={{
              backgroundColor: "rgba(255,255,255,0.1)",
              padding: "4px 12px",
              borderRadius: "20px",
              fontSize: "13px",
              backdropFilter: "blur(10px)",
              border: "1px solid rgba(255,255,255,0.1)",
            }}
          >
            {activeIndex + 1} / {items.length}
          </div>
        </div>
        <button
          onClick={(e) => {
            e.stopPropagation();
            onClose();
          }}
          style={{
            background: "rgba(255,255,255,0.1)",
            border: "1px solid rgba(255,255,255,0.2)",
            color: "white",
            width: "44px",
            height: "44px",
            borderRadius: "12px",
            cursor: "pointer",
            fontSize: "22px",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            transition: "all 0.2s cubic-bezier(0.4, 0, 0.2, 1)",
            outline: "none",
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.backgroundColor = "rgba(255, 59, 48, 0.8)";
            e.currentTarget.style.borderColor = "rgba(255, 59, 48, 0.2)";
            e.currentTarget.style.transform = "rotate(90deg)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.backgroundColor = "rgba(255,255,255,0.1)";
            e.currentTarget.style.borderColor = "rgba(255,255,255,0.2)";
            e.currentTarget.style.transform = "rotate(0deg)";
          }}
        >
          ✕
        </button>
      </div>

      {/* Main Content */}
      <div
        onClick={onClose}
        style={{
          flex: 1,
          width: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          position: "relative",
          cursor: "zoom-out",
        }}
      >
        {/* Loading Spinner */}
        {isLoading && (
          <div
            style={{
              position: "absolute",
              zIndex: 20,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: "15px",
            }}
          >
            <div
              className="preview-spinner"
              style={{
                width: "40px",
                height: "40px",
                border: "3px solid rgba(255,255,255,0.1)",
                borderTop: "3px solid #646cff",
                borderRadius: "50%",
                animation: "spin 1s linear infinite",
              }}
            />
            <style>{`
              @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
            `}</style>
          </div>
        )}

        {items.length > 1 && (
          <button
            disabled={activeIndex === 0}
            onClick={(e) => {
              e.stopPropagation();
              handlePrev();
            }}
            style={{
              position: "absolute",
              left: "40px",
              width: "60px",
              height: "60px",
              background: "rgba(255,255,255,0.05)",
              color: "white",
              border: "1px solid rgba(255,255,255,0.1)",
              borderRadius: "50%",
              cursor: activeIndex === 0 ? "default" : "pointer",
              opacity: activeIndex === 0 ? 0 : isLoading ? 0.3 : 1,
              transition: "all 0.3s",
              zIndex: 10,
              fontSize: "24px",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              backdropFilter: "blur(10px)",
            }}
            onMouseEnter={(e) => {
              if (activeIndex !== 0 && !isLoading) {
                e.currentTarget.style.backgroundColor =
                  "rgba(255,255,255,0.15)";
                e.currentTarget.style.transform = "scale(1.1)";
              }
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = "rgba(255,255,255,0.05)";
              e.currentTarget.style.transform = "scale(1)";
            }}
          >
            ‹
          </button>
        )}

        <div
          onClick={(e) => {
            e.stopPropagation();
          }}
          style={{
            maxWidth: "90vw",
            maxHeight: "85vh",
            boxShadow: "0 30px 60px rgba(0,0,0,0.8)",
            backgroundColor: "#000",
            cursor: "default",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            opacity: isLoading ? 0.5 : 1,
            transition: "opacity 0.2s",
          }}
        >
          {media.type === MediaType.MEDIA_IMAGE ? (
            <img
              src={currentUrl}
              onLoad={() => {
                setIsLoading(false);
              }}
              style={{
                maxWidth: "100%",
                maxHeight: "85vh",
                objectFit: "contain",
              }}
              alt="Preview"
            />
          ) : media.type === MediaType.MEDIA_VIDEO ? (
            <video
              key={currentUrl}
              src={currentUrl}
              onLoadedData={() => {
                setIsLoading(false);
              }}
              controls
              autoPlay
              style={{ maxWidth: "100%", maxHeight: "85vh" }}
            />
          ) : (
            <div
              style={{
                width: "400px",
                padding: "40px",
                backgroundColor: "#1a1a1a",
                borderRadius: "20px",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                gap: "30px",
                border: "1px solid rgba(255,255,255,0.1)",
              }}
            >
              <div
                style={{
                  width: "80px",
                  height: "80px",
                  borderRadius: "50%",
                  backgroundColor: "rgba(255,255,255,0.05)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: "40px",
                }}
              >
                🎵
              </div>
              <audio
                key={currentUrl}
                src={currentUrl}
                onLoadedData={() => {
                  setIsLoading(false);
                }}
                controls
                autoPlay
                style={{ width: "100%" }}
              />
            </div>
          )}
        </div>

        {items.length > 1 && (
          <button
            disabled={activeIndex === items.length - 1}
            onClick={(e) => {
              e.stopPropagation();
              handleNext();
            }}
            style={{
              position: "absolute",
              right: "40px",
              width: "60px",
              height: "60px",
              background: "rgba(255,255,255,0.05)",
              color: "white",
              border: "1px solid rgba(255,255,255,0.1)",
              borderRadius: "50%",
              cursor: activeIndex === items.length - 1 ? "default" : "pointer",
              opacity:
                activeIndex === items.length - 1 ? 0 : isLoading ? 0.3 : 1,
              transition: "all 0.3s",
              zIndex: 10,
              fontSize: "24px",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              backdropFilter: "blur(10px)",
            }}
            onMouseEnter={(e) => {
              if (activeIndex !== items.length - 1 && !isLoading) {
                e.currentTarget.style.backgroundColor =
                  "rgba(255,255,255,0.15)";
                e.currentTarget.style.transform = "scale(1.1)";
              }
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = "rgba(255,255,255,0.05)";
              e.currentTarget.style.transform = "scale(1)";
            }}
          >
            ›
          </button>
        )}
      </div>

      {/* Footer info */}
      <div
        style={{
          padding: "20px",
          color: "rgba(255,255,255,0.4)",
          fontSize: "13px",
          background: "linear-gradient(to top, rgba(0,0,0,0.5), transparent)",
          width: "100%",
          textAlign: "center",
          boxSizing: "border-box",
        }}
      >
        Use Left/Right arrow keys or side buttons to navigate. Press ESC to
        close.
      </div>
    </div>
  );
};
