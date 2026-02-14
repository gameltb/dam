import { useHotkeys } from "react-hotkeys-hook";

import { useGraphOperations } from "@/hooks/graph/useGraphOperations";
import { useFlowStore } from "@/store/flowStore";
import { useSettingsStore } from "@/store/ui/settingsStore";

export const useAppHotkeys = () => {
  const hotkeys = useSettingsStore((s) => s.hotkeys);
  const { redo, undo } = useFlowStore();
  const { autoLayout, copySelected, deleteSelected, duplicateSelected, paste } = useGraphOperations();

  useHotkeys(hotkeys.undo, (e) => {
    e.preventDefault();
    undo();
  });
  useHotkeys(hotkeys.redo, (e) => {
    e.preventDefault();
    redo();
  });
  useHotkeys(hotkeys.copy, (e) => {
    e.preventDefault();
    copySelected();
  });
  useHotkeys(hotkeys.paste, (e) => {
    e.preventDefault();
    paste();
  });
  useHotkeys(hotkeys.duplicate, (e) => {
    e.preventDefault();
    duplicateSelected();
  });
  useHotkeys(hotkeys.delete, (e) => {
    e.preventDefault();
    deleteSelected();
  });
  useHotkeys(hotkeys.autoLayout, (e) => {
    e.preventDefault();
    autoLayout();
  });
};
