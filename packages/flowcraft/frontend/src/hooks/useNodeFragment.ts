import { useFlowStore, type RFState } from "../store/flowStore";
import { useShallow } from "zustand/react/shallow";

/**
 * 极简的路径获取工具，支持 a.b.c 格式
 */
function getByPath(obj: unknown, path: string): unknown {
  if (!obj || typeof obj !== "object") return undefined;
  const parts = path.split(".");
  let current: unknown = obj;
  for (const part of parts) {
    if (current && typeof current === "object" && part in current) {
      current = (current as Record<string, unknown>)[part];
    } else {
      return undefined;
    }
  }
  return current;
}

/**
 * useNodeFragment 允许组件只订阅节点数据的特定子集。
 * 类似于 GraphQL Fragment，它能减少不必要的重渲染。
 *
 * @param nodeId 节点 ID
 * @param path 数据路径 (例如: 'data.label' 或 'data.widgets.slider_1.value')
 * @returns 路径对应的值
 */
// eslint-disable-next-line @typescript-eslint/no-unnecessary-type-parameters
export function useNodeFragment<T>(nodeId: string, path: string): T {
  // 显式定义 selector 的参数和返回类型，确保 T 被有效利用
  const selector = (state: RFState): T => {
    const node = state.nodes.find((n) => n.id === nodeId);
    const value = getByPath(node, path);
    return value as T;
  };

  return useFlowStore(useShallow(selector));
}
