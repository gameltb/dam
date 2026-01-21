import {
  create,
  type DescField,
  type DescMessage,
  fromJson,
  type Message,
  type MessageShape,
  toJson,
} from "@bufbuild/protobuf";
import { type AlgebraicType } from "spacetimedb";

/**
 * 能够提供 AlgebraicType 的包装对象
 */
interface HasAlgebraicType {
  algebraicType: AlgebraicType;
}

type StdbTypeArg = AlgebraicType | HasAlgebraicType;

/**
 * PB Object -> SpacetimeDB Storage Object
 */
export function pbToStdb(pbSchema: DescMessage, stdbType: StdbTypeArg, message: unknown): unknown {
  if (message === null || message === undefined) return null;

  if (isWkt(pbSchema.typeName)) {
    const finalized = isMessage(message) ? message : fromJson(pbSchema, message as any);
    return JSON.stringify(toJson(pbSchema, finalized));
  }

  const algType = extractAlgebraicType(stdbType);
  if (algType.tag !== "Product") {
    throw new Error(`[Bridge] Protocol mismatch for ${pbSchema.typeName}: Expected Product, got ${algType.tag}`);
  }

  const stdbElements = algType.value.elements;
  const pbObj = message as Record<string, any>;
  const result: Record<string, unknown> = {};

  for (const field of pbSchema.fields) {
    if (field.oneof) {
      const oneofValue = pbObj[field.oneof.localName];
      if (oneofValue?.case === field.localName) {
        const stdbElement = stdbElements.find((e) => e.name === field.oneof!.localName);
        if (!stdbElement) throw new Error(`[Bridge] Oneof column '${field.oneof.localName}' missing in STDB`);

        const oneofAlgType = extractAlgebraicType(stdbElement.algebraicType);
        if (oneofAlgType.tag !== "Sum") throw new Error(`[Bridge] Expected Sum for oneof '${field.oneof.localName}'`);

        const variant = oneofAlgType.value.variants.find((v) => v.name === field.localName);
        if (!variant) throw new Error(`[Bridge] Variant '${field.localName}' not found in Sum`);

        result[field.oneof.localName] = {
          tag: field.localName,
          value: transformValue(field, variant.algebraicType, oneofValue.value, true),
        };
      }
      continue;
    }

    const stdbElement = stdbElements.find((e) => e.name === field.localName);
    if (!stdbElement) continue;

    const val = pbObj[field.localName];
    if (val === undefined || val === null) {
      const targetAlgType = extractAlgebraicType(stdbElement.algebraicType);
      result[field.localName] = targetAlgType.tag === "Array" ? [] : undefined;
    } else {
      result[field.localName] = transformValue(field, stdbElement.algebraicType, val, true);
    }
  }
  return result;
}

/**
 * STDB Storage Object -> PB Object
 */
export function stdbToPb<T extends DescMessage>(pbSchema: T, stdbType: StdbTypeArg, stdbObj: unknown): MessageShape<T> {
  if (isWkt(pbSchema.typeName) && typeof stdbObj === "string") {
    return fromJson(pbSchema, JSON.parse(stdbObj)) as MessageShape<T>;
  }

  if (stdbObj === null || stdbObj === undefined) {
    return create(pbSchema) as MessageShape<T>;
  }

  const algType = extractAlgebraicType(stdbType);
  if (algType.tag !== "Product") {
    throw new Error(`[Bridge] Protocol mismatch for ${pbSchema.typeName}`);
  }

  const stdbElements = algType.value.elements;
  const rawStdb = stdbObj as Record<string, any>;
  const result: Record<string, any> = {};

  for (const field of pbSchema.fields) {
    if (field.oneof) {
      const snakeOneofName = field.oneof.name;
      const oneofData = (
        rawStdb[field.oneof.localName] !== undefined ? rawStdb[field.oneof.localName] : rawStdb[snakeOneofName]
      ) as undefined | { tag: string; value: unknown };

      if (oneofData?.tag === field.localName) {
        const stdbElement = stdbElements.find((e) => e.name === field.oneof!.localName);
        if (!stdbElement) throw new Error(`[Bridge] Oneof column '${field.oneof.localName}' missing in STDB`);

        const oneofAlgType = extractAlgebraicType(stdbElement.algebraicType);
        if (oneofAlgType.tag !== "Sum") throw new Error(`[Bridge] Expected Sum for oneof '${field.oneof.localName}'`);

        const variant = oneofAlgType.value.variants.find((v) => v.name === field.localName);
        if (!variant) throw new Error(`[Bridge] Variant '${field.localName}' not found in Sum`);

        result[field.oneof.localName] = {
          case: field.localName,
          value: transformValue(field, variant.algebraicType, oneofData.value, false),
        };
        // Also set the field name directly for better compatibility
        result[field.localName] = result[field.oneof.localName].value;
      }
      continue;
    }

    const stdbElement = stdbElements.find((e) => e.name === field.localName);
    if (!stdbElement) continue;

    const snakeName = field.name; // original proto name usually snake_case
    const stdbVal = rawStdb[field.localName] !== undefined ? rawStdb[field.localName] : rawStdb[snakeName];

    if (stdbVal !== undefined && stdbVal !== null) {
      result[field.localName] = transformValue(field, stdbElement.algebraicType, stdbVal, false);
    }
  }

  const finalResult = create(pbSchema, result as any) as MessageShape<T>;
  if (pbSchema.name === "NodeData" || pbSchema.name === "ChatMessage") {
    console.log(
      `[Bridge] Converted ${pbSchema.name}:`,
      JSON.stringify(result, (_, v) => (typeof v === "bigint" ? v.toString() : v)),
    );
  }
  return finalResult;
}

/**
 * 从 google.protobuf.Value 结构中提取原始 JS 值
 */
export function unwrapPbValue(v: any): any {
  if (v && typeof v === "object" && "kind" in v && v.kind) {
    const kind = v.kind;
    if (kind.case === "boolValue") return kind.value;
    if (kind.case === "numberValue") return kind.value;
    if (kind.case === "stringValue") return kind.value;
    if (kind.case === "nullValue") return null;
    if (kind.case === "structValue") return kind.value;
    if (kind.case === "listValue") return kind.value.values?.map(unwrapPbValue);
  }
  return v;
}

function extractAlgebraicType(arg: StdbTypeArg): AlgebraicType {
  const alg = (arg as HasAlgebraicType).algebraicType || (arg as AlgebraicType);
  if (alg.tag === "Sum" && alg.value.variants.length === 2) {
    const v = alg.value.variants;
    const n0 = v[0]?.name?.toLowerCase();
    const n1 = v[1]?.name?.toLowerCase();
    if ((n0 === "some" && n1 === "none") || (n0 === "none" && n1 === "some")) {
      const someVariant = n0 === "some" ? v[0] : v[1];
      if (someVariant) return extractAlgebraicType(someVariant.algebraicType);
    }
  }
  return alg;
}

function isMessage(val: unknown): val is Message {
  return val !== null && typeof val === "object" && "$typeName" in val;
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

  if (pbField.fieldKind === "enum" || (pbField.fieldKind === "list" && pbField.listKind === "enum")) {
    const pbEnum = pbField.enum;
    if (!pbEnum) throw new Error(`[Bridge] Missing enum descriptor for field ${(pbField as any).localName}`);
    if (toStdb) {
      const enumVal = pbEnum.values.find((v) => v.number === val);
      if (!enumVal) throw new Error(`[Bridge] Invalid enum value ${val} for ${pbEnum.name}`);
      return { tag: enumVal.name, value: {} };
    } else {
      const tag = (val as any).tag;
      if (!tag) throw new Error(`[Bridge] Missing tag in STDB enum for field ${(pbField as any).localName}`);
      const enumVal = pbEnum.values.find((v) => v.name === tag);
      if (!enumVal) throw new Error(`[Bridge] Invalid enum tag ${tag} for ${pbEnum.name}`);
      return enumVal.number;
    }
  }

  if (pbField.fieldKind === "message" || (pbField.fieldKind === "list" && pbField.listKind === "message")) {
    const pbMessage = pbField.message;
    if (!pbMessage) throw new Error(`[Bridge] Missing message descriptor for field ${(pbField as any).localName}`);
    if (toStdb) return pbToStdb(pbMessage, algType, val);
    return stdbToPb(pbMessage, algType, val);
  }

  // 严格的标量校验
  if (pbField.fieldKind === "scalar" && !toStdb) {
    const unwrapped = unwrapPbValue(val);
    if (pbField.scalar === 1 || pbField.scalar === 2) {
      // Double/Float
      const num = Number(unwrapped);
      if (isNaN(num)) {
        throw new Error(
          `[Bridge] Data Corruption: Field '${(pbField as any).localName}' is NaN. Raw value: ${JSON.stringify(val)}`,
        );
      }
      return num;
    }
    return unwrapped;
  }

  return val;
}

function transformValue(pbField: DescField, stdbType: AlgebraicType, val: unknown, toStdb: boolean): unknown {
  const algType = extractAlgebraicType(stdbType);

  if (pbField.fieldKind === "list") {
    if (algType.tag !== "Array" || !Array.isArray(val)) {
      throw new Error(`[Bridge] Type mismatch for list field ${(pbField as any).localName}`);
    }
    return val.map((item) => transformSingleValue(pbField, algType.value, item, toStdb));
  }

  if (pbField.fieldKind === "map") {
    if (algType.tag !== "Array") throw new Error(`[Bridge] Type mismatch for map field ${(pbField as any).localName}`);
    const entryStdbType = extractAlgebraicType(algType.value);
    if (entryStdbType.tag !== "Product") throw new Error(`[Bridge] Invalid map entry structure`);

    const keyElement = entryStdbType.value.elements.find((e) => e.name === "key");
    const valueElement = entryStdbType.value.elements.find((e) => e.name === "value");
    if (!keyElement || !valueElement) throw new Error(`[Bridge] Map missing elements`);

    if (toStdb) {
      const entries = isMessage(val)
        ? Object.entries(toJson(pbField.parent as any, val) as any)
        : Object.entries(val as Record<string, any>);
      return entries.map(([k, v]) => ({
        key: transformSingleValue(pbField, keyElement.algebraicType, k, true),
        value: transformSingleValue(pbField, valueElement.algebraicType, v, true),
      }));
    } else {
      if (!Array.isArray(val))
        throw new Error(`[Bridge] Expected array from STDB for map ${(pbField as any).localName}`);
      const result: Record<string, any> = {};
      val.forEach((entry) => {
        const k = transformSingleValue(pbField, keyElement.algebraicType, entry.key, false);
        const v = transformSingleValue(pbField, valueElement.algebraicType, entry.value, false);
        result[String(k)] = v;
      });
      return result;
    }
  }

  return transformSingleValue(pbField, algType, val, toStdb);
}
