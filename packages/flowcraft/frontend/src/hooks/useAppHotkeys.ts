import { useHotkeys } from "react-hotkeys-hook";
import { useShallow } from "zustand/react/shallow";

import { useFlowStore } from "@/store/flowStore";
import { useUiStore } from "@/store/uiStore";

import { useGraphOperations } from "./useGraphOperations";

/**
 * 全局应用快捷键管理
 * 绑定到 uiStore 中的动态配置，支持用户自定义
 */
export function useAppHotkeys() {
  const { redo, undo } = useFlowStore(
    useShallow((s) => ({
      redo: s.redo,
      undo: s.undo,
    })),
  );
  const { autoLayout, copySelected, deleteNode, duplicateSelected, paste } = useGraphOperations();

  // 从 store 获取用户设置的快捷键
  const hotkeys = useUiStore((s) => s.settings.hotkeys);

  // --- 历史管理 ---
  useHotkeys(hotkeys.undo, (e) => {
    e.preventDefault();
    undo();
  });

  useHotkeys(hotkeys.redo, (e) => {
    e.preventDefault();
    redo();
  });

  // --- 剪贴板与基本编辑 ---
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

  // 删除操作特殊处理：防止在输入框内触发
  useHotkeys(hotkeys.delete, (e) => {
    const target = e.target as HTMLElement;
    if (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable) {
      return;
    }

    const nodes = useFlowStore.getState().nodes;
    const selectedNodes = nodes.filter((n) => n.selected);
    selectedNodes.forEach((n) => {
      deleteNode(n.id);
    });
  });

  // --- 图表工具 ---
  useHotkeys(hotkeys.autoLayout, (e) => {
    e.preventDefault();
    autoLayout();
  });
}
