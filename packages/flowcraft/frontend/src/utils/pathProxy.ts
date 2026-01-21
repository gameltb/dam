/**
 * 创建一个针对具体对象的路径录制代理 (ORM 模式)
 *
 * 它会自动将 React Flow 的顶层属性 (width, height, position, selected)
 * 映射到符合 Protobuf Schema 的 presentation 嵌套路径。
 */
export function createNodeProxy<T extends object>(
  target: T,
  onCommit: (path: string, value: any) => void,
  path = "",
): T {
  const MAPPING: Record<string, string> = {
    data: "state", // AppNode.data -> Node.state
    height: "presentation.height",
    parentId: "presentation.parentId",
    position: "presentation.position",
    selected: "presentation.isSelected",
    width: "presentation.width",
  };

  return new Proxy(target, {
    get(t, prop: string) {
      if (typeof prop === "symbol" || prop.startsWith("$")) return (t as any)[prop];

      const val = (t as any)[prop];

      // 计算当前层级的物理路径名
      let segment = prop;
      if (!path && MAPPING[prop]) {
        segment = MAPPING[prop];
      }

      const nextPath = path ? `${path}.${segment}` : segment;

      if (val !== null && typeof val === "object") {
        return createNodeProxy(val, onCommit, nextPath);
      }

      if (val === undefined) {
        return createNodeProxy({} as any, onCommit, nextPath);
      }

      return val;
    },

    set(_t, prop: string, value: any) {
      if (typeof prop === "symbol") return false;

      let segment = prop;
      if (!path && MAPPING[prop]) {
        segment = MAPPING[prop];
      }

      const finalPath = path ? `${path}.${segment}` : segment;
      onCommit(finalPath, value);
      return true;
    },
  });
}
