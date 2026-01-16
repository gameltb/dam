# Protocol-Bridge 强类型架构方案 (V2.0)

本文档详细说明了 Flowcraft 项目中 `spacetime-module` 的 Schema 生成逻辑、对齐官方 SDK 的 Reducer 包装机制以及全链路强类型的实现细节。

## 1. 核心设计哲学

### 1.1 Protobuf 作为单源真理 (SSOT)

系统中的所有数据结构首先在 `schema/flowcraft/v1/` 的 `.proto` 文件中定义。

- **一致性**：后端数据库存储、Worker 逻辑和前端 UI 共享同一套类型描述符。
- **零 JSON 核心 (Zero-JSON Core)**：数据库表不再使用 `data_json` 字符串，而是映射为 SpacetimeDB 原生的嵌套 `Record` 和 `Enum`。

### 1.2 透明传输桥接 (Transparent Bridge)

通过生成的 `PbConnection` 包装器，实现前端对象与后端二进制流的无感转换：

- **调用端**：开发者直接调用 `conn.reducers.createNodePb({ node: nodeObj })`，参数为强类型 JS 对象。
- **传输层**：包装器利用 Protobuf 元数据自动执行 `toBinary`。
- **服务端**：`wrapPbHandler` 自动执行 `fromBinary`，业务逻辑直接处理解析后的对象。

---

## 2. 自动化构建流水线

构建逻辑由 `flowcraft.config.yaml` 统一配置，包含以下关键环节：

### 2.1 零 IO Schema 生成 (`proto-to-stdb.ts`)

该脚本通过 `buf build` 捕获二进制描述符流，直接在内存中解析并映射为 SpacetimeDB 类型。

- **Oneof 支持**：自动将 Protobuf 的 `oneof` 展开为 SpacetimeDB 的 `t.enum` 和可选字段组合。
- **循环依赖处理**：自动识别递归引用（如 `google.protobuf.Value`）并降级为 `t.string()` 存储，确保数据库引擎稳定性。

### 2.2 智能客户端生成 (`generate-pb-client.ts`)

生成对齐官方 SDK 规范的 `PbConnection` 包装类：

- **命名对齐**：自动将后端 `snake_case` 方法名和参数名转换为前端 `camelCase`。
- **多文件内省**：自动识别类型所属的 `.proto` 文件，生成精确的 `@/generated/...` 导入语句。
- **类型合并**：利用 TypeScript 交叉类型，使包装器既具备官方 `DbConnection` 的功能，又拥有增强后的强类型 Reducers。

---

## 3. 开发规范

### 3.1 定义强类型表

在 `spacetime-module/src/tables/*.ts` 中，直接引用生成的描述符：

```typescript
import { t, table } from "spacetimedb/server";
import { Node } from "../generated/generated_schema";

export const nodes = table(
  { name: "nodes", public: true },
  {
    node_id: t.string().primaryKey(),
    state: Node, // 引用生成的强类型结构
    // ...
  },
);
```

### 3.2 配置文件 (`flowcraft.config.yaml`)

所有生成路径均在此定义，禁止在脚本中硬编码：

```yaml
stdb_schema:
  output_path: "spacetime-module/src/generated/generated_schema.ts"
pb_client:
  reducers_dir: "spacetime-module/src/reducers"
  output_path: "src/generated/pb_client.ts"
```

---

## 4. 质量保障 (CI/CD)

所有构建脚本必须通过 `npm run scripts:check`，包含：

- **严格类型检查**：通过 `tsconfig.scripts.json` 约束。
- **定制化 Lint**：针对 Mock 逻辑和动态解析优化的 ESLint 规则。
