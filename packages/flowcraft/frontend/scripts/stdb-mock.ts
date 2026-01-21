import { createRequire } from "module";

/**
 * 核心：支持链式调用的类型描述符 Mock
 * 增加 __ts_type 用于在生成代码时还原 TypeScript 类型
 */
const createTypeMock = (kind: string, tsType: string, name?: string) => {
  const mock: any = {
    __st_kind: kind,
    __st_name: name,
    __ts_type: tsType,
    autoIncrement: () => mock,
    indexed: () => mock,
    name: name, // For generate-pb-client compatibility
    primaryKey: () => mock,
    unique: () => mock,
  };
  return mock;
};

export const mockT: any = {
  array: (inner: any) => ({
    ...createTypeMock("array", `${inner?.__ts_type ?? "any"}[]`),
    __st_inner: inner,
  }),
  bool: () => createTypeMock("bool", "boolean"),
  byteArray: () => createTypeMock("byteArray", "Uint8Array"),
  enum: (name: string, _obj: any) => createTypeMock("enum", name, name),
  f32: () => createTypeMock("number", "number"),
  f64: () => createTypeMock("number", "number"),
  i32: () => createTypeMock("number", "number"),
  i64: () => createTypeMock("bigint", "bigint"),
  object: (name: string, _obj: any) => createTypeMock("object", name, name),
  option: (inner: any) => ({
    ...inner,
    __st_optional: true,
    __ts_type: `${inner.__ts_type} | undefined`,
  }),
  string: () => createTypeMock("string", "string"),
  u32: () => createTypeMock("number", "number"),
  u64: () => createTypeMock("bigint", "bigint"),
  unit: () => createTypeMock("unit", "void"),
};

export function setupStdbMock(capturedTables: any[]) {
  const require = createRequire(import.meta.url);
  const Module = require("module");
  const originalLoad = Module._load;

  Module._load = function (request: string, _parent: any, _isMain: boolean) {
    if (request === "spacetimedb/server") {
      return {
        schema: () => ({ reducer: () => {} }),
        spacetimedb: { reducer: () => {} },
        t: mockT,
        table: (meta: any, schema: any) => {
          capturedTables.push({ name: meta.name, schema });
          return { rowType: {} };
        },
      };
    }
    if (request.startsWith("spacetime:sys")) return {};

    if (request.includes("generated_schema")) {
      return new Proxy(
        {},
        {
          get: (_target, prop: string) => {
            if (prop === "__esModule") return true;
            // 默认认为从 schema 导出的都是 object
            return createTypeMock("object", prop, prop);
          },
        },
      );
    }

    return originalLoad(request, _parent, _isMain);
  };

  return () => {
    Module._load = originalLoad;
  };
}
