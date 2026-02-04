import { create, type DescMessage, type MessageShape, toBinary as pbToBinary, toJson } from "@bufbuild/protobuf";
import { type GenMessage } from "@bufbuild/protobuf/codegenv2";

import { PB_REDUCERS_MAP, TABLE_TO_PROTO } from "@/generated/pb_metadata";
import { type DbConnection } from "@/generated/spacetime";
import { NodeKernel } from "@/kernel/NodeKernel";

import { stdbToPb } from "./proto-stdb-bridge";

// Keep compatibility aliases
export type PbClient = PbConnection;

/**
 * Wrapped connector type
 */
export type PbConnection = DbConnection & {
  convertStdbToPb: typeof convertStdbToPb;
  kernel: NodeKernel;
  pbreducers: ProjectedReducers;
};

/**
 * Type utility: Generate PB-enhanced Reducers types based on metadata mapping
 */
export type PbReducersProjection<R, M> = {
  [K in keyof R]: R[K] extends (params: infer P) => void
    ? (params: TransformPbParams<P, K extends keyof M ? M[K] : Record<string, never>>) => void
    : R[K];
};

/**
 * Type utility: Replace specific fields in the parameter object with PB message types
 */
export type TransformPbParams<P, PbMapping> = {
  [K in keyof P]: K extends keyof PbMapping
    ? PbMapping[K] extends { schema: GenMessage<infer S> }
      ? S extends DescMessage
        ? MessageShape<S>
        : unknown
      : unknown
    : P[K];
};

/**
 * Core type: Automatically generate PB-enhanced Reducers signatures through projection
 */
type ProjectedReducers = PbReducersProjection<DbConnection["reducers"], typeof PB_REDUCERS_MAP>;

/**
 * Converts an STDB Row to a standard PB object.
 * Cross-table aggregation logic removed, reverted to atomic component mapping.
 * @param _db Deprecated, no longer need to manually handle view joins
 */
export function convertStdbToPb(tableName: string, row: any, _db?: any): any {
  const meta = (TABLE_TO_PROTO as any)[tableName];
  if (!meta) return row; // If no mapping exists, return the original row directly

  const rawVal = row[meta.field];
  const pbObj = stdbToPb(meta.schema, rawVal);

  // Core Improvement: All mapped tables must go through create() to ensure default values (e.g., empty arrays) exist.
  // This ensures that the FE won't crash when accessing .length.
  return create(meta.schema, pbObj as any);
}

/**
 * Converts an STDB Row to a pure JSON object (for debug display).
 * If a PB mapping exists, convert it to JSON format (enum names, including default values).
 */
export function convertStdbToPbJson(tableName: string, row: any): any {
  const meta = (TABLE_TO_PROTO as any)[tableName];
  if (!meta) return row;

  // 1. Convert to PB Message
  const msg = convertStdbToPb(tableName, row);

  // 2. Convert Message to JSON
  // useProtoFieldName: false -> camelCase keys (standard for JSON)
  // emitDefaultValues: true -> show all fields
  // enumAsInteger: false -> show Enum names (readable)
  try {
    return toJson(meta.schema, msg, {
      emitDefaultValues: true,
      enumAsInteger: false,
      useProtoFieldName: false,
    });
  } catch (e) {
    console.warn(`[PbClient] JSON serialization failed for table ${tableName}`, e);
    return row;
  }
}

/**
 * Creates PB proxy implementation logic
 */
export function createPbProxy(
  target: Record<string, unknown>,
  pbMetadata: Record<string, Record<string, { schema: any }>>,
): any {
  return new Proxy(target, {
    get(t, prop: string) {
      const original = t[prop];
      if (typeof original !== "function") return original;

      const fieldMapping = pbMetadata[prop];
      if (!fieldMapping) return original;

      return (params: Record<string, unknown>) => {
        const wrapped = { ...params };
        for (const [field, meta] of Object.entries(fieldMapping)) {
          const val = wrapped[field];
          if (val && typeof val === "object" && !(val instanceof Uint8Array)) {
            try {
              const msg = create(meta.schema, val);
              wrapped[field] = pbToBinary(meta.schema, msg);
            } catch (e) {
              console.error(`[PbClient] Serialization failed for ${prop}.${field}:`, e);
            }
          }
        }
        return original.call(t, wrapped);
      };
    },
  });
}

/**
 * Core wrapping function: Upgrades DbConnection to a version supporting automatic PB serialization
 */
export function wrapReducers(conn: DbConnection): PbConnection {
  const proxy = createPbProxy(conn.reducers, PB_REDUCERS_MAP);
  const wrapped = Object.assign(conn, { pbreducers: proxy }) as any;
  wrapped.kernel = new NodeKernel(wrapped);
  // Bind conversion function
  wrapped.convertStdbToPb = (tableName: string, row: any) => convertStdbToPb(tableName, row);
  return wrapped as PbConnection;
}
