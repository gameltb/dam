import {
  create,
  type DescEnum,
  type DescField,
  type DescMessage,
  fromJsonString,
  type Message,
  type MessageShape,
  toJsonString,
} from "@bufbuild/protobuf";
import { reflect } from "@bufbuild/protobuf/reflect";
import { type AlgebraicType } from "spacetimedb";

/**
 * 能够提供 AlgebraicType 的包装对象（如 ProductBuilder, SumBuilder）
 */
interface HasAlgebraicType {
  algebraicType: AlgebraicType;
}

/**
 * 允许传入原始代数类型或 Builder 对象
 */
type StdbTypeArg = AlgebraicType | HasAlgebraicType;

/**
 * PB Object -> SpacetimeDB Storage Object
 */
export function pbToStdb(pbSchema: DescMessage, stdbType: StdbTypeArg, message: unknown): unknown {
  if (message === null || message === undefined) return null;

  if (isWkt(pbSchema.typeName)) {
    return toJsonString(pbSchema, message as Message);
  }

  // Ensure message is a valid Message instance
  let msgInstance = message as Message | Record<string, unknown>;
  if (msgInstance && typeof msgInstance === "object" && "message" in msgInstance && "desc" in msgInstance) {
    // It seems we received a ReflectMessageImpl, unwrap it
    msgInstance = (msgInstance as unknown as { message: Message }).message;
  }

  // reflect() requires a Message instance, but nested fields from create() with plain objects might be plain objects.
  const finalizedInstance = create(pbSchema, msgInstance as MessageShape<DescMessage>);

  const rMessage = reflect(pbSchema, finalizedInstance);
  const result: Record<string, unknown> = {};

  const algType = extractAlgebraicType(stdbType);
  if (algType.tag !== "Product") {
    throw new Error(`Expected Product type for message ${pbSchema.typeName}, got ${algType.tag}`);
  }

  const stdbElements = algType.value.elements;

  for (const field of pbSchema.fields) {
    const stdbElement = stdbElements.find((e) => e.name === field.localName);
    if (!stdbElement) continue;

    const targetStdbType = stdbElement.algebraicType;

    if (!rMessage.isSet(field)) {
      const targetAlgType = extractAlgebraicType(targetStdbType);
      result[field.localName] = targetAlgType.tag === "Array" ? [] : undefined;
      continue;
    }

    const val = rMessage.get(field);

    if (field.oneof) {
      const selected = rMessage.oneofCase(field.oneof);
      if (selected === field) {
        // Use { tag, value } format for SumTypes (Oneofs)
        result[field.oneof.localName] = {
          tag: field.localName,
          value: transformValue(field, targetStdbType, val, true),
        };
      }
      continue;
    }

    result[field.localName] = transformValue(field, targetStdbType, val, true);
  }
  return result;
}

/**
 * STDB Storage Object -> PB Object
 */
export function stdbToPb<T extends DescMessage>(pbSchema: T, stdbType: StdbTypeArg, stdbObj: unknown): MessageShape<T> {
  if (isWkt(pbSchema.typeName) && typeof stdbObj === "string") {
    return fromJsonString(pbSchema, stdbObj) as MessageShape<T>;
  }

  const pbObj = create(pbSchema) as MessageShape<T>;
  if (stdbObj === null || stdbObj === undefined) return pbObj;

  const algType = extractAlgebraicType(stdbType);
  if (algType.tag !== "Product") {
    throw new Error(`Expected Product type for message ${pbSchema.typeName}, got ${algType.tag}`);
  }

  const stdbElements = algType.value.elements;
  const rawStdb = stdbObj as Record<string, unknown>;

  for (const field of pbSchema.fields) {
    const stdbElement = stdbElements.find((e) => e.name === field.localName);
    if (!stdbElement) continue;

    const targetStdbType = stdbElement.algebraicType;
    const stdbVal = rawStdb[field.localName];

    if (field.oneof) {
      const oneofObj = rawStdb[field.oneof.localName] as undefined | { tag: string; value: unknown };
      // Handle { tag, value } format for Oneofs
      if (oneofObj?.tag === field.localName) {
        (pbObj as Record<string, unknown>)[field.oneof.localName] = {
          case: field.localName,
          value: transformValue(field, targetStdbType, oneofObj.value, false),
        };
      }
      continue;
    }

    if (stdbVal === undefined || stdbVal === null) continue;
    (pbObj as Record<string, unknown>)[field.localName] = transformValue(field, targetStdbType, stdbVal, false);
  }
  return pbObj;
}

/**
 * 辅助函数：安全地提取原始 AlgebraicType
 */
function extractAlgebraicType(arg: StdbTypeArg): AlgebraicType {
  const alg = (arg as HasAlgebraicType).algebraicType || (arg as AlgebraicType);

  // 处理 t.option 包装
  if (alg.tag === "Sum" && alg.value.variants.length === 2) {
    const v = alg.value.variants;
    const v0 = v[0];
    const v1 = v[1];
    if (v0?.name === "some" && v1?.name === "none") {
      return extractAlgebraicType(v0.algebraicType);
    }
  }
  return alg;
}

function isWkt(typeName: string): boolean {
  return (
    typeName === "google.protobuf.Struct" ||
    typeName === "google.protobuf.Value" ||
    typeName === "google.protobuf.ListValue"
  );
}

function transformSingleValue(pbField: DescField, stdbType: AlgebraicType, val: unknown, toStdb: boolean): unknown {
  const algType = extractAlgebraicType(stdbType);

  // Handle Enums
  if (pbField.fieldKind === "enum" || (pbField.fieldKind === "list" && pbField.listKind === "enum")) {
    const pbEnum = (pbField as { enum: DescEnum }).enum;
    if (!pbEnum) return val;

    if (toStdb) {
      const enumVal = pbEnum.values.find((v) => v.number === val);
      if (!enumVal) throw new Error(`Enum value ${String(val)} not found for ${pbField.name}`);
      return { tag: enumVal.name, value: {} };
    } else {
      const variantName = (val as { tag: string }).tag;
      const enumVal = pbEnum.values.find((v) => v.name === variantName);
      if (!enumVal) throw new Error(`Enum tag ${variantName} not found for ${pbField.name}`);
      return enumVal.number;
    }
  }

  // Handle Messages
  if (pbField.fieldKind === "message" || (pbField.fieldKind === "list" && pbField.listKind === "message")) {
    const pbMessage = (pbField as { message: DescMessage }).message;
    if (!pbMessage) return val;

    if (isWkt(pbMessage.typeName) && algType.tag === "String") {
      let valToUse = val;
      if (valToUse && typeof valToUse === "object" && "message" in valToUse && "desc" in valToUse) {
        valToUse = (valToUse as unknown as { message: Message }).message;
      }
      return toStdb ? toJsonString(pbMessage, valToUse as Message) : fromJsonString(pbMessage, val as string);
    }
    return toStdb ? pbToStdb(pbMessage, algType, val) : stdbToPb(pbMessage, algType, val);
  }

  return val;
}

function transformValue(pbField: DescField, stdbType: AlgebraicType, val: unknown, toStdb: boolean): unknown {
  if (val === null || val === undefined) {
    if (toStdb && pbField.fieldKind === "list") return [];
    return val;
  }

  const algType = extractAlgebraicType(stdbType);

  // 1. Handle Lists
  if (pbField.fieldKind === "list") {
    if (algType.tag !== "Array") {
      throw new Error(`Type mismatch: PB field '${pbField.name}' is a list, but STDB type is '${algType.tag}'`);
    }
    const list = Array.from(val as Iterable<unknown>);
    const innerStdbType = algType.value;
    return list.map((item) => transformSingleValue(pbField, innerStdbType, item, toStdb));
  }

  // 2. Map Support
  if (pbField.fieldKind === "map") {
    if (algType.tag !== "Array") {
      throw new Error(`Type mismatch: PB field '${pbField.name}' is a map, but STDB type is '${algType.tag}'`);
    }
    const entryStdbType = algType.value;
    const unwrappedEntryType = extractAlgebraicType(entryStdbType);
    if (unwrappedEntryType.tag !== "Product") {
      throw new Error(`Expected Product for map entry, got ${unwrappedEntryType.tag}`);
    }
    const keyType = unwrappedEntryType.value.elements.find((e) => e.name === "key")?.algebraicType;
    const valueType = unwrappedEntryType.value.elements.find((e) => e.name === "value")?.algebraicType;

    if (!keyType || !valueType) {
      throw new Error(`Map entry product missing key or value fields`);
    }

    if (toStdb) {
      const map = val as Map<unknown, unknown> | Record<string, unknown>;
      const result: unknown[] = [];
      const entries =
        typeof (map as Iterable<unknown>)[Symbol.iterator] === "function"
          ? Array.from(map as Iterable<[unknown, unknown]>)
          : Object.entries(map as Record<string, unknown>);

      for (const [key, value] of entries) {
        if (key === "$typeName") continue;
        result.push({
          key: transformSingleValue(pbField, keyType, key, true),
          value: transformSingleValue(pbField, valueType, value, true),
        });
      }
      return result;
    } else {
      const entries = val as { key: unknown; value: unknown }[];
      const result: Record<string, unknown> = {};
      for (const entry of entries) {
        const key = transformSingleValue(pbField, keyType, entry.key, false);
        const value = transformSingleValue(pbField, valueType, entry.value, false);
        result[String(key)] = value;
      }
      return result;
    }
  }

  return transformSingleValue(pbField, algType, val, toStdb);
}