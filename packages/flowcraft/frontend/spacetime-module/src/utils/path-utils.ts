import { type DescMessage } from "@bufbuild/protobuf";
import { type Value } from "@bufbuild/protobuf/wkt";

/**
 * 从 google.protobuf.Value 结构中提取原始 JS 值
 */
export function unwrapPbValue(v: unknown): unknown {
  if (v && typeof v === "object" && "kind" in v) {
    const val = v as Value;
    if (!val.kind) return v;
    
    if (val.kind.case === "boolValue") return val.kind.value;
    if (val.kind.case === "numberValue") return val.kind.value;
    if (val.kind.case === "stringValue") return val.kind.value;
    if (val.kind.case === "nullValue") return null;
    if (val.kind.case === "structValue") return val.kind.value;
    if (val.kind.case === "listValue") return val.kind.value.values.map(unwrapPbValue);
  }
  return v;
}

/**
 * 递归应用路径更新到普通 JS 对象 (纯函数实现)
 * 增加 schema 参数以支持 oneof 自动转换
 */
export function applyPathToObj(
  obj: any, 
  parts: string[], 
  value: any, 
  updateType: number,
  schema?: DescMessage
): any {
  if (parts.length === 0) return value;
  
  const [first, ...rest] = parts;
  if (!first) return value;

  const currentObj = obj || {};
  let targetKey = first;
  let targetValue = value;
  let nextSchema: DescMessage | undefined;

  // 1. Oneof 自动映射逻辑
  if (schema) {
    const field = schema.fields.find(f => f.name === first || f.jsonName === first);
    if (field) {
      if (field.oneof) {
        // 如果该字段属于某个 oneof (如 extension)，需要特殊处理
        targetKey = field.oneof.localName;
        // 如果是最后一层，或者下一层不是 oneof 映射的结果，
        // 则我们需要在这里包装成 { case, value }
        if (rest.length === 0) {
           targetValue = { case: field.localName, value: value };
        } else {
           // 递归处理：我们将 value 注入到当前的 oneof 容器中
           const existingOneof = currentObj[targetKey] || {};
           const subValue = applyPathToObj(
             existingOneof.case === (field.localName as any) ? existingOneof.value : {},
             rest, 
             value, 
             updateType, 
             field.fieldKind === "message" ? field.message : undefined
           );
           targetValue = { case: field.localName, value: subValue };
           // 已经处理完 rest 了，不需要继续外层的逻辑
           return { ...currentObj, [targetKey]: targetValue };
        }
      } else {
        if (field.fieldKind === "message") nextSchema = field.message;
      }
    }
  }

  // 2. 普通路径逻辑
  // 如果是最后一层路径
  if (rest.length === 0) {
    if (updateType === 2) { // DELETE
      const { [targetKey]: _, ...newObj } = currentObj;
      return newObj;
    }
    // REPLACE / MERGE
    return {
      ...currentObj,
      [targetKey]: targetValue
    };
  }

  // 递归处理嵌套层级
  const currentLevelValue = (currentObj[targetKey]) || {};
  return {
    ...currentObj,
    [targetKey]: applyPathToObj(currentLevelValue, rest, targetValue, updateType, nextSchema)
  };
}