import { useTheme } from "../hooks/useTheme";

type ContextMenuProps = {
  x: number;
  y: number;
  onClose: () => void;
  onDelete?: () => void;
  onFocus?: () => void;
  onShowDamEntity?: () => void;
  onToggleTheme?: () => void;
  onAddTextNode?: () => void;
  onAddImageNode?: () => void;
  isPaneMenu?: boolean;
  dynamicActions?: { id: string; name: string; onClick: () => void }[];
};

export function ContextMenu({
  x,
  y,
  onClose,
  onDelete,
  onFocus,
  onShowDamEntity,
  onToggleTheme,
  onAddTextNode,
  onAddImageNode,
  isPaneMenu = false,
  dynamicActions = [],
}: ContextMenuProps) {
  const { theme } = useTheme();

  const menuStyle: React.CSSProperties = {
    position: "absolute",
    top: y,
    left: x,
    backgroundColor: theme === "dark" ? "#2a2a2a" : "white",
    color: theme === "dark" ? "#f0f0f0" : "#213547",
    border: `1px solid ${theme === "dark" ? "#444" : "#ddd"}`,
    borderRadius: 5,
    zIndex: 1000,
    boxShadow: "0 2px 5px rgba(0,0,0,0.15)",
  };

  const menuItemStyle: React.CSSProperties = {
    padding: "8px 20px",
    cursor: "pointer",
  };

  const separatorStyle: React.CSSProperties = {
    borderTop: `1px solid ${theme === "dark" ? "#444" : "#ddd"}`,
    margin: "4px 0",
  };

  return (
    <div style={menuStyle} onClick={onClose}>
      <ul style={{ listStyle: "none", margin: 0, padding: "5px 0" }}>
        {!isPaneMenu && onDelete && (
          <li onClick={onDelete} style={menuItemStyle}>
            Delete
          </li>
        )}
        {!isPaneMenu && onFocus && (
          <li onClick={onFocus} style={menuItemStyle}>
            Focus
          </li>
        )}
        {!isPaneMenu && onShowDamEntity && (
          <li onClick={onShowDamEntity} style={menuItemStyle}>
            显示DAM实体
          </li>
        )}

        {dynamicActions.length > 0 && <div style={separatorStyle} />}
        {dynamicActions.map((action) => (
          <li key={action.id} onClick={action.onClick} style={menuItemStyle}>
            {action.name}
          </li>
        ))}

        {(onToggleTheme || onAddTextNode || onAddImageNode) && (
          <div style={separatorStyle} />
        )}

        <div style={{ padding: "5px 10px", fontSize: "0.8em", color: "#888" }}>
          Debug
        </div>

        {onToggleTheme && (
          <li onClick={onToggleTheme} style={menuItemStyle}>
            Switch Theme
          </li>
        )}
        {onAddTextNode && (
          <li onClick={onAddTextNode} style={menuItemStyle}>
            Add Text Node
          </li>
        )}
        {onAddImageNode && (
          <li onClick={onAddImageNode} style={menuItemStyle}>
            Add Image Node
          </li>
        )}
      </ul>
    </div>
  );
}
