import { useMemo } from "react";
import { useTheme } from "../hooks/useTheme";
import { type NodeTemplate } from "../types";

type ContextMenuProps = {
  x: number;
  y: number;
  onClose: () => void;
  onDelete?: () => void;
  onDeleteEdge?: () => void;
  onFocus?: () => void;
  onOpenEditor?: () => void;
  onToggleTheme?: () => void;
  templates?: NodeTemplate[];
  onAddNode?: (template: NodeTemplate) => void;
  onAutoLayout?: () => void;
  onGroupSelected?: () => void;
  onLayoutGroup?: () => void;
  onGalleryAction?: (url: string) => void;
  galleryItemUrl?: string;
  isPaneMenu?: boolean;
  dynamicActions?: { id: string; name: string; onClick: () => void }[];
};

interface MenuTree {
  [key: string]: {
    template?: NodeTemplate;
    children?: MenuTree;
  };
}

export function ContextMenu({
  x,
  y,
  onClose,
  onDelete,
  onDeleteEdge,
  onFocus,
  onOpenEditor,
  onToggleTheme,
  templates = [],
  onAddNode,
  onAutoLayout,
  onGroupSelected,
  onLayoutGroup,
  onGalleryAction,
  galleryItemUrl,
  isPaneMenu = false,
  dynamicActions = [],
}: ContextMenuProps) {
  const { theme } = useTheme();
  const isDark = theme === "dark";

  // Build the hierarchical tree from flat templates
  const templateTree = useMemo(() => {
    const tree: MenuTree = {};
    templates.forEach((tpl) => {
      let current = tree;
      tpl.path.forEach((part) => {
        if (!current[part]) {
          current[part] = { children: {} };
        }
        current = current[part].children!;
      });
      current[tpl.label] = { template: tpl };
    });
    return tree;
  }, [templates]);

  const menuStyle: React.CSSProperties = {
    position: "absolute",
    top: y,
    left: x,
    backgroundColor: isDark ? "#2a2a2a" : "white",
    color: isDark ? "#f0f0f0" : "#213547",
    border: `1px solid ${isDark ? "#444" : "#ddd"}`,
    borderRadius: 5,
    zIndex: 1000,
    boxShadow: "0 2px 10px rgba(0,0,0,0.2)",
    minWidth: "180px",
    padding: "5px 0",
  };

  const menuItemStyle: React.CSSProperties = {
    padding: "8px 16px",
    cursor: "pointer",
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    fontSize: "13px",
    position: "relative",
  };

  const separatorStyle: React.CSSProperties = {
    borderTop: `1px solid ${isDark ? "#444" : "#ddd"}`,
    margin: "4px 0",
  };

  return (
    <div
      style={menuStyle}
      onMouseLeave={onClose}
      onClick={(e) => e.stopPropagation()}
    >
      <ul style={{ listStyle: "none", margin: 0, padding: 0 }}>
        {galleryItemUrl && onGalleryAction && (
          <li
            onClick={() => {
              onGalleryAction(galleryItemUrl);
              onClose();
            }}
            style={{ ...menuItemStyle, fontWeight: "bold", color: "#646cff" }}
            className="menu-item"
          >
            ✨ Create Node from Image
          </li>
        )}
        {onDeleteEdge && (
          <li
            onClick={() => {
              onDeleteEdge();
              onClose();
            }}
            style={menuItemStyle}
            className="menu-item"
          >
            Disconnect
          </li>
        )}
        {!isPaneMenu && onDelete && (
          <li
            onClick={() => {
              onDelete();
              onClose();
            }}
            style={menuItemStyle}
            className="menu-item"
          >
            Delete
          </li>
        )}
        {!isPaneMenu && onFocus && (
          <li
            onClick={() => {
              onFocus();
              onClose();
            }}
            style={menuItemStyle}
            className="menu-item"
          >
            Focus
          </li>
        )}
        {!isPaneMenu && onOpenEditor && (
          <li
            onClick={() => {
              onOpenEditor();
              onClose();
            }}
            style={menuItemStyle}
            className="menu-item"
          >
            Open Editor
          </li>
        )}

        {dynamicActions.length > 0 && <div style={separatorStyle} />}
        {dynamicActions.map((action) => (
          <li
            key={action.id}
            onClick={() => {
              action.onClick();
              onClose();
            }}
            style={menuItemStyle}
            className="menu-item"
          >
            {action.name}
          </li>
        ))}

        {(onAutoLayout || onGroupSelected || onLayoutGroup) && (
          <div style={separatorStyle} />
        )}
        {onAutoLayout && (
          <li
            onClick={() => {
              onAutoLayout();
              onClose();
            }}
            style={menuItemStyle}
            className="menu-item"
          >
            Auto Layout
          </li>
        )}
        {onGroupSelected && (
          <li
            onClick={() => {
              onGroupSelected();
              onClose();
            }}
            style={menuItemStyle}
            className="menu-item"
          >
            Group Selected
          </li>
        )}
        {onLayoutGroup && (
          <li
            onClick={() => {
              onLayoutGroup();
              onClose();
            }}
            style={menuItemStyle}
            className="menu-item"
          >
            Layout Group
          </li>
        )}

        <div style={separatorStyle} />

        <li style={{ ...menuItemStyle, cursor: "default", opacity: 0.6 }}>
          Add Node
        </li>

        <RecursiveMenu
          tree={templateTree}
          onAddNode={(tpl) => {
            onAddNode?.(tpl);
            onClose();
          }}
          isDark={isDark}
        />

        <div style={separatorStyle} />

        {onToggleTheme && (
          <li
            onClick={() => {
              onToggleTheme();
              onClose();
            }}
            style={menuItemStyle}
            className="menu-item"
          >
            Switch Theme
          </li>
        )}
      </ul>
      <style>{`
        .menu-item:hover { background-color: ${isDark ? "#3a3a3a" : "#f5f5f5"}; }
        .menu-parent:hover > .submenu { display: block !important; }
      `}</style>
    </div>
  );
}

function RecursiveMenu({
  tree,
  onAddNode,
  isDark,
}: {
  tree: MenuTree;
  onAddNode: (tpl: NodeTemplate) => void;
  isDark: boolean;
}) {
  return (
    <>
      {Object.entries(tree).map(([key, value]) => {
        const hasChildren =
          value.children && Object.keys(value.children).length > 0;

        return (
          <li
            key={key}
            style={{
              padding: "8px 16px",
              cursor: "pointer",
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              fontSize: "13px",
              position: "relative",
            }}
            className={hasChildren ? "menu-parent" : "menu-item"}
            onClick={() => {
              if (value.template) {
                onAddNode(value.template);
              }
            }}
          >
            <span>{hasChildren ? `📁 ${key}` : `📄 ${key}`}</span>
            {hasChildren && <span>▶</span>}

            {hasChildren && (
              <div
                className="submenu"
                style={{
                  display: "none",
                  position: "absolute",
                  top: 0,
                  left: "100%",
                  backgroundColor: isDark ? "#2a2a2a" : "white",
                  border: `1px solid ${isDark ? "#444" : "#ddd"}`,
                  borderRadius: 5,
                  boxShadow: "0 2px 10px rgba(0,0,0,0.2)",
                  minWidth: "160px",
                  padding: "5px 0",
                  zIndex: 1001,
                }}
              >
                <ul style={{ listStyle: "none", margin: 0, padding: 0 }}>
                  <RecursiveMenu
                    tree={value.children!}
                    onAddNode={onAddNode}
                    isDark={isDark}
                  />
                </ul>
              </div>
            )}
          </li>
        );
      })}
    </>
  );
}
