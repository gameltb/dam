# Flowcraft Protobuf & 数据库设计规范 (V1.1)

本文档定义了 Flowcraft 项目中 Protobuf 消息、SpacetimeDB 表结构以及前端状态管理的三位一体设计策略。

## 1. 结构分层原则 (Layering Strategy)

所有业务对象必须严格划分为三个逻辑层，禁止跨层混用字段：

| 层次                      | 职责                                | 字段示例               | 存储位置                          |
| :------------------------ | :---------------------------------- | :--------------------- | :-------------------------------- |
| **Identity (身份层)**     | 唯一标识、关系引用、查询索引        | `node_id`, `parent_id` | SpacetimeDB 顶层列 (PK/Indexed)   |
| **Presentation (表现层)** | 视觉状态（客户端主控，后端透传）    | `position`, `width`    | `Node.presentation` (嵌套 Record) |
| **Domain (领域层)**       | 业务逻辑核心（Worker/Reducer 修改） | `tree_id`, `extension` | `NodeData` (嵌套 Record)          |

## 2. 命名与语义规范 (Naming & Semantics)

通过后缀明确消息的生命周期：

- **`...State`**: 持久化存储对象。代表“事实”，存入数据库。
- **`...Request`**: **指令模型 (Command Model)**。代表“意图”，作为 Reducer 输入。
- **`...Response`**: 结果反馈。
- **`...Event`**: 异步通知。用于流式数据或临时总线。

## 3. 指令模型与元数据组合 (Mutation Pattern)

为了实现类似“继承”的特征，所有改变图状态的指令必须包含 `MutationMetadata`：

```proto
// 在 core/base.proto 定义
message MutationMetadata {
  string origin_task_id = 1; // 溯源 ID
  int64 timestamp = 2;
  MutationSource source = 3;
}

// 具体指令实现
message AddNodeRequest {
  MutationMetadata metadata = 1; // 必须放在 1 号位
  Node node = 2;
}
```

## 4. RJSF 映射策略 (RJSF Strategy)

为了保证 `react-jsonschema-form` 生成的 UI 易用且直观：

- **扁平化设计**：业务配置字段尽量保持扁平。嵌套层级超过 3 层会导致 UI 极其混乱。
- **强类型优先**：尽量避免使用 `google.protobuf.Struct`。如果字段类型固定（如 Slider 的数值），应在 `extension` 中使用显式类型。
- **默认值占位**：所有 Enum 必须有 `_UNSPECIFIED = 0`，防止 RJSF 默认选中错误的业务选项。

## 5. 目录与包名规范 (Organization)

- **语义路径**：`core/` (基础), `nodes/` (节点状态), `actions/` (执行参数), `services/` (通讯)。
- **点号包名**：`package flowcraft.v1.[module];` 必须与物理路径对应。
- **显式导入**：必须从根目录开始导入，如 `import "flowcraft/v1/core/base.proto";`。

## 7. 前端开发最佳实践 (Frontend Patterns)

### 7.1 纯粹消息传递 (Pure Message Passing)

系统已废弃 `Mutate` 包装工厂。现在直接使用生成的 Request 类构建指令，并依赖 Protobuf v2 的元数据进行自动映射。

**推荐用法：**

```typescript
import { create as createProto } from "@bufbuild/protobuf";
import { AddNodeRequestSchema, UpdateNodeRequestSchema } from "@/generated/flowcraft/v1/core/service_pb";

applyMutations([
  createProto(AddNodeRequestSchema, { node: myNode }),
  createProto(UpdateNodeRequestSchema, { nodeId: "n1", presentation: { width: 100 } }),
]);
```

**优势：**

1. **零逻辑包装**：不再需要维护手写的映射函数，完全由元数据驱动。
2. **原生类型收窄**：利用 `switch (input.$typeName)` 即可获得完美的 TypeScript 详尽性检查。
3. **透明序列化**：通过 `SchemaRegistry` 自动实现日志审计和网络同步。
