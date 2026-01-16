import { toBinary as pbToBinary, type MessageShape, type DescMessage } from "@bufbuild/protobuf";
import { type GenMessage } from "@bufbuild/protobuf/codegenv2";
import { stdbToPb } from "./proto-stdb-bridge";

/**
 * 类型工具：将参数对象中的特定字段替换为 PB 消息类型
 */
export type TransformPbParams<P, PbMapping> = {
  [K in keyof P]: K extends keyof PbMapping 
    ? (PbMapping[K] extends { schema: GenMessage<infer S> } 
        ? (S extends DescMessage ? MessageShape<S> : any) 
        : any)
    : P[K];
};

/**
 * 类型工具：基于元数据映射生成 PB 增强版 Reducers 类型
 */
export type PbReducersProjection<R, M> = {
  [K in keyof R]: R[K] extends (params: infer P) => void
    ? (params: TransformPbParams<P, K extends keyof M ? M[K] : {}>) => void
    : R[K];
};

/**
 * 创建 PB 代理实现逻辑
 */
export function createPbProxy(
  target: any,
  pbMetadata: Record<string, Record<string, { schema: any }>>
) {
  return new Proxy(target, {
    get(t, prop: string) {
      const original = t[prop];
      if (typeof original !== 'function') return original;

      const fieldMapping = pbMetadata[prop];
      if (!fieldMapping) return original;

      return (params: any) => {
        const wrapped = { ...params };
        for (const [field, meta] of Object.entries(fieldMapping)) {
          const val = wrapped[field];
          if (val && typeof val === 'object' && !(val instanceof Uint8Array)) {
            try {
              wrapped[field] = pbToBinary(meta.schema as any, val);
            } catch (e) {
              console.error(`[PbClient] Serialization failed for ${prop}.${field}:`, e);
            }
          }
        }
        return original(wrapped);
      };
    }
  });
}

/**
 * 核心转换工具实现
 * 根据用户调试观察到的结构：rowType.elements[i].name 和 algebraicType
 */
export function convertStdbToPbInternal(
  tableName: string, 
  row: any, 
  tablesMap: any, 
  tableToProto: Record<string, { schema: any, field: string }>
): any {
  const meta = tableToProto[tableName];
  if (!meta) throw new Error("No mapping found for table " + tableName);
  
  const table = tablesMap[tableName];
  if (!table) throw new Error("No table accessor found for " + tableName);

  const rowType = table.rowType;
  if (!rowType) {
    throw new Error(`No rowType found for table ${tableName}`);
  }

  // 根据观察，elements 可能直接在 rowType 上，也可能在 rowType.value 上
  const elements = rowType.elements || rowType.value?.elements;
  if (!elements) {
    throw new Error(`Could not find elements in rowType of table ${tableName}`);
  }

  const colElement = elements.find((e: any) => e.name === meta.field);
  if (!colElement) {
    throw new Error(`Field ${meta.field} not found in elements of table ${tableName}`);
  }

  // 使用找到的 colElement.algebraicType 进行转换
  return stdbToPb(meta.schema, colElement.algebraicType, row[meta.field]);
}

/**
 * 包装连接器实现
 */
export function wrapReducersInternal(conn: any, pbMetadata: any): any {
  const proxy = createPbProxy(conn.reducers, pbMetadata);
  return Object.assign(conn, { pbreducers: proxy });
}
