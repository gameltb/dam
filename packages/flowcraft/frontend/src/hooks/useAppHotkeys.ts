import { useHotkeys } from "react-hotkeys-hook";
import { useShallow } from "zustand/react/shallow";

import { useFlowStore, useTemporalStore } from "@/store/flowStore";
import { useUiStore } from "@/store/uiStore";
import { type AppNode, type Edge } from "@/types";

import { useGraphOperations } from "./useGraphOperations";

export const useAppHotkeys = () => {
  const { autoLayout, copySelected, deleteEdge, deleteNode, duplicateSelected, paste } = useGraphOperations();

  const { edges, nodes } = useFlowStore(
    useShallow((state) => ({
      edges: state.edges,
      nodes: state.nodes,
    })),
  );

  const { redo, undo } = useTemporalStore(
    useShallow((state) => ({
      redo: state.redo,
      undo: state.undo,
    })),
  );

  const hotkeys = useUiStore((s) => s.settings.hotkeys);

  useHotkeys(
    hotkeys.copy,
    (e) => {
      e.preventDefault();
      copySelected();
    },
    { enableOnFormTags: false },
  );

  useHotkeys(
    hotkeys.paste,
    (e) => {
      e.preventDefault();
      paste();
    },
    { enableOnFormTags: false },
  );

  useHotkeys(
    hotkeys.duplicate,
    (e) => {
      e.preventDefault();
      duplicateSelected();
    },
    { enableOnFormTags: false },
  );

  useHotkeys(
    hotkeys.autoLayout,
    (e) => {
      e.preventDefault();
      autoLayout();
    },
    { enableOnFormTags: false },
  );

  useHotkeys(
    hotkeys.undo,
    (e) => {
      e.preventDefault();
      undo();
    },
    { enableOnFormTags: false },
  );

  useHotkeys(
    hotkeys.redo,
    (e) => {
      e.preventDefault();
      redo();
    },
    { enableOnFormTags: false },
  );

  useHotkeys(
    hotkeys.delete,
    (e) => {
      e.preventDefault();
      const selectedNodes = nodes.filter((n: AppNode) => n.selected);
      const selectedEdges = edges.filter((e: Edge) => e.selected);
      selectedNodes.forEach((n: AppNode) => {
        deleteNode(n.id);
      });
      selectedEdges.forEach((e: Edge) => {
        deleteEdge(e.id);
      });
    },
    { enableOnFormTags: false },
  );
};
