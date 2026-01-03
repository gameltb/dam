import { useHotkeys } from "react-hotkeys-hook";
import { useUiStore } from "../store/uiStore";
import { useGraphOperations } from "./useGraphOperations";
import { useFlowStore, useTemporalStore } from "../store/flowStore";
import { useShallow } from "zustand/react/shallow";
import { type AppNode, type Edge } from "../types";

export const useAppHotkeys = () => {
  const {
    copySelected,
    paste,
    duplicateSelected,
    autoLayout,
    deleteNode,
    deleteEdge,
  } = useGraphOperations({
    clientVersion: 0,
  });

  const { nodes, edges } = useFlowStore(
    useShallow((state) => ({
      nodes: state.nodes,
      edges: state.edges,
    })),
  );

  const { undo, redo } = useTemporalStore(
    useShallow((state) => ({
      undo: state.undo,
      redo: state.redo,
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
