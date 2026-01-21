import { type DescMessage } from "@bufbuild/protobuf";

export interface SchemaDraftOptions {
  basePath?: string;
  pathMapper?: (prop: string) => null | string;
}

/**
 * A type-safe proxy for Protobuf messages that records path updates.
 */
export function createSchemaDraft<T extends object>(
  target: T,
  schema: DescMessage,
  onCommit: (path: string, value: unknown) => void,
  options: SchemaDraftOptions = {},
): T {
  const { basePath = "", pathMapper } = options;

  return new Proxy(target, {
    get(t: T, prop: string | symbol): unknown {
      if (typeof prop !== "string" || prop.startsWith("$")) {
        return Reflect.get(t, prop);
      }

      const val = Reflect.get(t, prop) as unknown;
      const mappedProp = pathMapper?.(prop) || prop;

      // 1. Try to find as a regular field
      const field = schema.fields.find((f) => f.localName === mappedProp || f.name === mappedProp);

      // 2. Check if it's a oneof (like 'extension')
      if (!field) {
        const oo = schema.oneofs.find((o) => o.localName === mappedProp || o.name === mappedProp);
        if (oo && val !== null && typeof val === "object" && "case" in val && "value" in val) {
          const variantName = (val as { case: string }).case;
          const variantValue = (val as { value: object }).value;
          const variantField = schema.fields.find((f) => f.oneof === oo && f.localName === variantName);

          if (variantField && variantField.fieldKind === "message") {
            const currentPath = basePath ? `${basePath}.${variantField.name}` : variantField.name;
            // Return a proxy directly into the oneof's value, but using the variant name in the path
            return createSchemaDraft(variantValue, variantField.message, onCommit, {
              basePath: currentPath,
            });
          }
        }
      }

      const physicalName = field ? field.name : mappedProp;
      const currentPath = basePath ? `${basePath}.${physicalName}` : physicalName;

      if (val !== null && typeof val === "object" && !(val instanceof Uint8Array)) {
        const nextSchema = field?.fieldKind === "message" ? field.message : schema;
        return createSchemaDraft(val, nextSchema as any, onCommit, {
          basePath: currentPath,
        });
      }

      if (val === undefined && field?.fieldKind === "message") {
        return createSchemaDraft({} as object, field.message, onCommit, {
          basePath: currentPath,
        });
      }

      return val;
    },

    set(t: T, prop: string | symbol, value: unknown): boolean {
      if (typeof prop !== "string") return false;

      const mappedProp = pathMapper?.(prop) || prop;
      const field = schema.fields.find((f) => f.localName === mappedProp || f.name === mappedProp);
      const physicalName = field ? field.name : mappedProp;
      const finalPath = basePath ? `${basePath}.${physicalName}` : physicalName;

      Reflect.set(t, prop, value);
      onCommit(finalPath, value);
      return true;
    },
  });
}
