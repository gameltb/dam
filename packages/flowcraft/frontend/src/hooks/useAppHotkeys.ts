import { useHotkeys } from "react-hotkeys-hook";
import { useUiStore } from "../store/uiStore";
import { useGraphOperations } from "./useGraphOperations";
import { useTemporalStore } from "../store/flowStore";
import { useShallow } from "zustand/react/shallow";

export const useAppHotkeys = () => {
  const { copySelected, paste, duplicateSelected, autoLayout } =
    useGraphOperations({
      clientVersion: 0,
    });

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
};
