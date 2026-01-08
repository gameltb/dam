import { type MenuNode } from "../components/base/SubMenu";
import { type NodeTemplate } from "@/types";

export const buildNodeTree = (templates: NodeTemplate[]): MenuNode[] => {
  const menuTree: MenuNode[] = [];
  templates.forEach((tpl) => {
    let currentLevel = menuTree;
    tpl.menuPath.forEach((part) => {
      let node = currentLevel.find((n) => n.label === part);
      if (!node) {
        node = { children: [], label: part };
        currentLevel.push(node);
      }
      currentLevel = node.children ?? [];
    });
    currentLevel.push({ label: tpl.displayName, template: tpl });
  });
  return menuTree;
};

export const buildActionTree = (
  dynamicActions: {
    id: string;
    name: string;
    onClick: () => void;
    path?: string[];
  }[],
): MenuNode[] => {
  const actionTree: MenuNode[] = [];
  dynamicActions.forEach((action) => {
    // Use provided path or split name by /
    const parts = action.path ? [...action.path] : action.name.split("/");
    const leafName = action.path ? action.name : (parts.pop() ?? "Action");
    let currentLevel = actionTree;

    parts.forEach((part) => {
      let node = currentLevel.find((n) => n.label === part);
      if (!node) {
        node = { children: [], label: part };
        currentLevel.push(node);
      }
      currentLevel = node.children ?? [];
    });
    currentLevel.push({
      action: { ...action, name: leafName },
      label: leafName,
    });
  });
  return actionTree;
};
