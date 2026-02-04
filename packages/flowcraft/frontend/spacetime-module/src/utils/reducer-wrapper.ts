import { type DescMessage, fromBinary } from "@bufbuild/protobuf";
import { type ReducerCtx } from "spacetimedb/server";

/**
 * Creates a PB deserialization wrapper for a single Reducer.
 * Used by the backend only, responsible for restoring binary parameters to standard PB objects.
 */
export function wrapPbHandler<P extends Record<string, any> = any>(
  args: Record<string, unknown>,
  handler: (ctx: ReducerCtx<any>, params: P) => void,
): (ctx: ReducerCtx<any>, params: P) => void {
  const pbFields: Record<string, DescMessage> = {};
  for (const [key, type] of Object.entries(args)) {
    if (type && typeof type === "object" && "typeName" in type) {
      pbFields[key] = type as DescMessage;
    }
  }

  return (ctx: ReducerCtx<any>, params: P) => {
    const finalParams = { ...params } as any;
    for (const [key, schema] of Object.entries(pbFields)) {
      const val = params[key];
      if (val instanceof Uint8Array) {
        // Keep the original binary reference to avoid subsequent toBinary calls
        finalParams[`${key}Binary`] = val;
        // Deserialize into an object for logic use
        finalParams[key] = fromBinary(schema, val);
      }
    }
    handler(ctx, finalParams);
  };
}
