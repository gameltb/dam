import { type NodeTemplate } from "../types";
import { type MenuNode } from "../components/base/SubMenu";

export const buildNodeTree = (templates: NodeTemplate[]): MenuNode[] => {
  const menuTree: MenuNode[] = [];
  templates.forEach((tpl) => {
    let currentLevel = menuTree;
    tpl.path.forEach((part) => {
      let node = currentLevel.find((n) => n.label === part);
      if (!node) {
        node = { label: part, children: [] };
        currentLevel.push(node);
      }
      currentLevel = node.children ?? [];
    });
    currentLevel.push({ label: tpl.label, template: tpl });
  });
  return menuTree;
};

export const buildActionTree = (
  dynamicActions: { id: string; name: string; onClick: () => void }[],
): MenuNode[] => {
  const actionTree: MenuNode[] = [];
  dynamicActions.forEach((action) => {
    const parts = action.name.split("/");
    const leafName = parts.pop() ?? "Action";
    let currentLevel = actionTree;

    parts.forEach((part) => {
      let node = currentLevel.find((n) => n.label === part);
      if (!node) {
        node = { label: part, children: [] };
        currentLevel.push(node);
      }
      currentLevel = node.children ?? [];
    });
    currentLevel.push({
      label: leafName,
      action: { ...action, name: leafName },
    });
  });
  return actionTree;
};
