import { type DescMessage, fromBinary } from "@bufbuild/protobuf";
import { type ReducerCtx } from "spacetimedb/server";

import { type AppSchema } from "../schema";

/**
 * 为单个 Reducer 创建 PB 反序列化包装器
 * 仅后端使用，负责将二进制参数还原为标准 PB 对象
 */
export function wrapPbHandler<S extends AppSchema, P extends Record<string, any>>(
  args: Record<string, unknown>,
  handler: (ctx: ReducerCtx<S>, params: P) => void,
) {
  const pbFields: Record<string, DescMessage> = {};
  for (const [key, type] of Object.entries(args)) {
    if (type && typeof type === "object" && "typeName" in type) {
      pbFields[key] = type as DescMessage;
    }
  }

  return (ctx: ReducerCtx<S>, params: P) => {
    const finalParams = { ...params } as any;
    for (const [key, schema] of Object.entries(pbFields)) {
      if (params[key] instanceof Uint8Array) {
        finalParams[key] = fromBinary(schema, params[key]);
      }
    }
    handler(ctx, finalParams);
  };
}
