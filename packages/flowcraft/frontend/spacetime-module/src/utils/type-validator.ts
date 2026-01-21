import { type DescMessage, type DescEnum, type DescField } from "@bufbuild/protobuf";
import { unwrapPbValue } from "./path-utils";

/**
 * 验证路径更新的值类型是否匹配 Protobuf 定义
 */
export function validateValueByPath(schema: DescMessage, path: string[], value: unknown): void {
  // 使用统一的解包逻辑进行校验
  const rawValue = unwrapPbValue(value);
  
  let current: DescMessage | DescEnum = schema;

  for (let i = 0; i < path.length; i++) {
    const part = path[i];
    if (!part) continue;

    if (current.kind !== "message") {
      throw new Error(`[Validator] Cannot traverse path '${part}' on a non-message type`);
    }

    const field: DescField | undefined = current.fields.find(f => f.name === part || f.jsonName === part);
    
    if (!field) {
      throw new Error(`[Validator] Field '${part}' not found in message ${current.name}`);
    }

    if (i === path.length - 1) {
      checkType(field, rawValue);
      return;
    }

    if (field.fieldKind === "message") {
      current = field.message;
    } else {
      throw new Error(`[Validator] Field '${part}' is a terminal type ${field.fieldKind}`);
    }
  }
}

function checkType(field: DescField, value: unknown): void {
  if (value === null || value === undefined) return;

  switch (field.fieldKind) {
    case "scalar": {
      const scalarType = field.scalar; 
      if (scalarType === 9) { // TYPE_STRING
        if (typeof value !== "string") throw TypeError(`Expected string, got ${typeof value}`);
      } else if ([1, 2, 3, 4, 5, 13].includes(scalarType)) { // Numeric types
        if (typeof value !== "number" && typeof value !== "bigint") throw TypeError(`Expected number, got ${typeof value}`);
      } else if (scalarType === 8) { // TYPE_BOOL
        if (typeof value !== "boolean") throw TypeError(`Expected boolean, got ${typeof value}`);
      }
      break;
    }
    case "enum":
      if (typeof value !== "number") throw TypeError(`Expected enum (number), got ${typeof value}`);
      break;
    case "message":
      if (typeof value !== "object") throw TypeError(`Expected object/message, got ${typeof value}`);
      break;
  }
}
