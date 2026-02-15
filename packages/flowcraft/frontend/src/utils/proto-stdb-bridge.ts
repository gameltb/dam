import { create, type DescMessage, fromBinary, fromJson, type MessageShape } from "@bufbuild/protobuf";

/**
 * STDB Storage -> PB Object
 * Simplified version: only supports Binary and WKT JSON strings.
 */
export function stdbToPb<T extends DescMessage>(pbSchema: T, stdbObj: unknown): MessageShape<T> {
  if (stdbObj === null || stdbObj === undefined) {
    return create(pbSchema);
  }

  // 1. Binary to PB
  if (stdbObj instanceof Uint8Array) {
    return fromBinary(pbSchema, stdbObj);
  }

  // 2. JSON String (WKT) to PB
  if (isWkt(pbSchema.typeName) && typeof stdbObj === "string") {
    try {
      return fromJson(pbSchema, JSON.parse(stdbObj)) as MessageShape<T>;
    } catch {
      return create(pbSchema);
    }
  }

  // 3. Pass-through for primitive types (number, string, boolean, bigint)
  if (
    typeof stdbObj === "number" ||
    typeof stdbObj === "string" ||
    typeof stdbObj === "boolean" ||
    typeof stdbObj === "bigint"
  ) {
    return stdbObj as unknown as MessageShape<T>;
  }

  // 4. Handle SpacetimeDB Enum wrapping (e.g., { mode: 1 } -> 1)
  if (typeof stdbObj === "object" && stdbObj !== null && !Array.isArray(stdbObj) && !(stdbObj instanceof Uint8Array)) {
    const keys = Object.keys(stdbObj);
    const firstKey = keys[0];
    if (keys.length === 1 && firstKey !== undefined) {
      const val = (stdbObj as Record<string, unknown>)[firstKey];
      if (typeof val === "number") {
        return val as unknown as MessageShape<T>;
      }
    }
  }

  return stdbObj as MessageShape<T>;
}

function isWkt(typeName: string): boolean {
  return (
    typeName === "google.protobuf.Struct" ||
    typeName === "google.protobuf.Value" ||
    typeName === "google.protobuf.ListValue"
  );
}
