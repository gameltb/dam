import { execSync } from "child_process";
import * as fs from "fs";

import { loadConfig } from "./config-loader";

async function main() {
  const config = loadConfig().stdb_schema;
  const OUTPUT_PATH = config.output_path;

  const imageJson = execSync("npx buf build schema -o -#format=json", {
    encoding: "utf8",
    maxBuffer: 10 * 1024 * 1024,
  });
  const image = JSON.parse(imageJson);

  let output = `/** AUTO-GENERATED - DO NOT EDIT **/
/* eslint-disable */
import { type Infer, t } from "spacetimedb/server";

export const Value = t.string();
export const Struct = t.string();
export const ListValue = t.string();

`;

  const messages: any[] = [];
  const enums: any[] = [];

  function collect(file: any) {
    if (file.enumType) {
      file.enumType.forEach((en: any) => {
        en._fullName = en.name;
        enums.push(en);
      });
    }
    if (file.messageType) {
      file.messageType.forEach((msg: any) => {
        collectMessage(msg, "");
      });
    }
  }

  function collectMessage(msg: any, parentName: string) {
    const fullName = parentName ? `${parentName}_${msg.name}` : msg.name;
    msg._fullName = fullName;
    messages.push(msg);

    if (msg.enumType) {
      msg.enumType.forEach((en: any) => {
        en._fullName = `${fullName}_${en.name}`;
        enums.push(en);
      });
    }
    if (msg.nestedType) {
      msg.nestedType.forEach((nested: any) => {
        collectMessage(nested, fullName);
      });
    }
  }

  image.file?.forEach((file: any) => {
    if (file.name?.startsWith("google/protobuf")) return;
    collect(file);
  });

  const typeMap: Record<string, string> = {
    TYPE_BOOL: "t.bool()",
    TYPE_BYTES: "t.byteArray()",
    TYPE_DOUBLE: "t.f64()",
    TYPE_FIXED32: "t.u32()",
    TYPE_FIXED64: "t.u64()",
    TYPE_FLOAT: "t.f32()",
    TYPE_INT32: "t.i32()",
    TYPE_INT64: "t.i64()",
    TYPE_STRING: "t.string()",
    TYPE_UINT32: "t.u32()",
    TYPE_UINT64: "t.u64()",
  };

  enums.forEach((en) => {
    output += `export const ${en._fullName} = t.enum("${en._fullName}", {
`;
    const values = en.value ?? [];

    values.forEach((v: any) => {
      output += `  ${v.name}: t.unit(),
`;
    });
    output += `});
export type ${en._fullName} = Infer<typeof ${en._fullName}>;

`;
  });

  // Sort messages to resolve dependencies
  const sortedMessages: any[] = [];
  const visited = new Set<string>();
  const visiting = new Set<string>();

  function getDeps(msg: any): string[] {
    const deps: string[] = [];
    msg.field?.forEach((f: any) => {
      if (f.type === "TYPE_MESSAGE" || f.type === "TYPE_ENUM") {
        const parts = f.typeName.split(".").filter((p: string) => !!p && p !== "flowcraft_proto" && p !== "v1");
        // Try suffixes from longest to shortest
        for (let i = 0; i < parts.length; i++) {
          const candidate = parts.slice(i).join("_");
          if (candidate !== msg._fullName) {
            deps.push(candidate);
            break;
          }
        }
      }
    });
    return deps;
  }

  function visit(msgName: string) {
    if (visited.has(msgName)) return;
    if (visiting.has(msgName)) return; // Circular or self-ref

    visiting.add(msgName);
    const msg = messages.find((m) => m._fullName === msgName);
    if (msg) {
      getDeps(msg).forEach((dep) => {
        if (messages.some((m) => m._fullName === dep)) {
          visit(dep);
        }
      });
      sortedMessages.push(msg);
    }
    visiting.delete(msgName);
    visited.add(msgName);
  }

  messages.forEach((m) => {
    visit(m._fullName);
  });

  const processed = new Set(enums.map((e) => e._fullName));
  processed.add("Value");
  processed.add("Struct");
  processed.add("ListValue");

  sortedMessages.forEach((msg) => {
    output += `// --- ${msg._fullName} ---
`;
    if (msg.oneofDecl) {
      msg.oneofDecl.forEach((oneof: any, index: number) => {
        const enumName = `${msg._fullName}_${oneof.name}`;
        output += `export const ${enumName} = t.enum("${enumName}", {
`;
        msg.field
          ?.filter((f: any) => f.oneofIndex === index)
          .forEach((f: any) => {
            const fieldName = f.jsonName ?? f.name;
            output += `  ${fieldName}: ${resolveSafeType(f, msg._fullName, processed)},
`;
          });
        output += `});
export type ${enumName} = Infer<typeof ${enumName}>;

`;
      });
    }

    output += `export const ${msg._fullName} = t.object("${msg._fullName}", {
`;
    msg.field?.forEach((f: any) => {
      if (f.oneofIndex !== undefined) return;
      const fieldName = f.jsonName ?? f.name;
      let type = resolveSafeType(f, msg._fullName, processed);

      const isComplex = f.type === "TYPE_MESSAGE" || f.type === "TYPE_ENUM";

      if (f.label === "LABEL_REPEATED") {
        // Repeated fields are mapped to t.array
        const elementType = isComplex ? `t.option(${type})` : type;
        type = `t.array(${elementType})`;
      } else if (isComplex) {
        // Wrap complex types in option to allow null/undefined (typical in PB)
        type = `t.option(${type})`;
      }

      output += `  ${fieldName}: ${type},
`;
    });
    msg.oneofDecl?.forEach((o: any) => {
      const fieldName = o.jsonName ?? o.name;
      // Oneofs are always optional in SpacetimeDB schema since they can be unset in PB
      output += `  ${fieldName}: t.option(${msg._fullName}_${o.name}),
    `;
    });

    output += `});
export type ${msg._fullName} = Infer<typeof ${msg._fullName}>;

`;
    processed.add(msg._fullName);
  });

  output += `
export const SCHEMA_METADATA = {
  messages: {
`;
  sortedMessages.forEach((msg) => {
    output += `    "${msg._fullName}": ${msg._fullName},
`;
  });
  output += `  },
  enums: {
`;
  enums.forEach((en) => {
    output += `    "${en._fullName}": ${en._fullName},
`;
  });
  output += `  }
} as const;
`;

  function resolveSafeType(f: any, currentMsg: string, done: Set<string>) {
    if (f.type === "TYPE_MESSAGE" || f.type === "TYPE_ENUM") {
      const parts = f.typeName.split(".").filter((p: string) => !!p && p !== "flowcraft_proto" && p !== "v1");

      // Try suffixes from longest to shortest
      for (let i = 0; i < parts.length; i++) {
        const candidate = parts.slice(i).join("_");
        if (done.has(candidate)) {
          return candidate;
        }
      }

      const candidate = parts.join("_");
      if (candidate === currentMsg) {
        return "t.string()";
      }
      return candidate; // Fallback
    }
    return typeMap[f.type];
  }

  fs.writeFileSync(OUTPUT_PATH, output);
  console.log("✅ generated.");
}

main().catch(console.error);
