import { useReactFlow } from "@xyflow/react";
import { ChevronDown, ChevronLeft, ChevronRight, ChevronUp } from "lucide-react";
import { memo, useCallback, useEffect } from "react";

import { cn } from "@/lib/utils";
import { useFlowStore } from "@/store/flowStore";
import { useNavigationStore } from "@/store/ui/navigationStore";
import { useUiStore } from "@/store/uiStore";
import { localToGlobal } from "@/utils/coordinateUtils";

interface NodeJumpOverlayProps {
  id: string;
  isEditing?: boolean;
}

export const NodeJumpOverlay = memo(({ id, isEditing }: NodeJumpOverlayProps) => {
  const relations = useFlowStore(useCallback((s) => s.nodeRelations[id], [id]));
  const navigatingNodeId = useUiStore((s) => s.navigatingNodeId);
  const setNavigatingNode = useUiStore((s) => s.setNavigatingNode);
  const setLastMousePos = useUiStore((s) => s.setLastMousePos);
  const rf = useReactFlow();

  const isActiveFocus = navigatingNodeId === id;

  const jumpTo = useCallback(
    (targetId: string) => {
      const latestState = useFlowStore.getState();
      const targetNode = latestState.nodesById[targetId];
      const currentScopeId = useNavigationStore.getState().activeScopeId;

      if (!targetNode) return;

      const performJump = (jumpId: string) => {
        const node = useFlowStore.getState().nodesById[jumpId];
        if (!node) return;

        const globalPos = localToGlobal(node.position, node.parentId || null, useFlowStore.getState().nodesById);
        const x = globalPos.x + (node.measured?.width ?? 300) / 2;
        const y = globalPos.y + (node.measured?.height ?? 150) / 2;

        rf.setCenter(x, y, { duration: 200, zoom: rf.getZoom() });

        setNavigatingNode(jumpId, true);
        if (window.lastProcessedMousePos) {
          setLastMousePos(window.lastProcessedMousePos);
        }
      };

      if ((targetNode.scopeId || null) !== currentScopeId) {
        useNavigationStore.getState().setActiveScope(targetNode.scopeId || null);
        setTimeout(() => {
          performJump(targetId);
        }, 150);
      } else {
        performJump(targetId);
      }
    },
    [rf, setNavigatingNode, setLastMousePos],
  );

  useEffect(() => {
    if (!isActiveFocus || isEditing) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (!relations) return;

      switch (e.key) {
        case "ArrowDown":
          if (relations.firstChildId) {
            e.preventDefault();
            jumpTo(relations.firstChildId);
          }
          break;
        case "ArrowLeft":
          if (relations.prevSiblingId) {
            e.preventDefault();
            jumpTo(relations.prevSiblingId);
          }
          break;
        case "ArrowRight":
          if (relations.nextSiblingId) {
            e.preventDefault();
            jumpTo(relations.nextSiblingId);
          }
          break;
        case "ArrowUp":
          if (relations.parentId) {
            e.preventDefault();
            jumpTo(relations.parentId);
          }
          break;
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [isActiveFocus, isEditing, relations, jumpTo]);

  if (!relations) return null;

  return (
    <div
      className={cn(
        "absolute inset-0 pointer-events-none z-50 transition-opacity duration-200",
        isActiveFocus ? "opacity-100" : "opacity-0 group-hover/node:opacity-100",
      )}
    >
      {!!relations.parentId && (
        <button
          className="absolute -top-8 left-1/2 -translate-x-1/2 p-1 bg-primary text-primary-foreground rounded-full pointer-events-auto hover:scale-110 transition-transform shadow-lg"
          onClick={(e) => {
            e.stopPropagation();
            jumpTo(relations.parentId!);
          }}
          title="Jump Up"
        >
          <ChevronUp size={16} />
        </button>
      )}
      {!!relations.firstChildId && (
        <button
          className="absolute -bottom-8 left-1/2 -translate-x-1/2 p-1 bg-primary text-primary-foreground rounded-full pointer-events-auto hover:scale-110 transition-transform shadow-lg"
          onClick={(e) => {
            e.stopPropagation();
            jumpTo(relations.firstChildId!);
          }}
          title="Jump Down"
        >
          <ChevronDown size={16} />
        </button>
      )}
      {!!relations.prevSiblingId && (
        <button
          className="absolute top-1/2 -left-8 -translate-y-1/2 p-1 bg-primary text-primary-foreground rounded-full pointer-events-auto hover:scale-110 transition-transform shadow-lg"
          onClick={(e) => {
            e.stopPropagation();
            jumpTo(relations.prevSiblingId!);
          }}
          title="Jump Left"
        >
          <ChevronLeft size={16} />
        </button>
      )}
      {!!relations.nextSiblingId && (
        <button
          className="absolute top-1/2 -right-8 -translate-y-1/2 p-1 bg-primary text-primary-foreground rounded-full pointer-events-auto hover:scale-110 transition-transform shadow-lg"
          onClick={(e) => {
            e.stopPropagation();
            jumpTo(relations.nextSiblingId!);
          }}
          title="Jump Right"
        >
          <ChevronRight size={16} />
        </button>
      )}
    </div>
  );
});

NodeJumpOverlay.displayName = "NodeJumpOverlay";
