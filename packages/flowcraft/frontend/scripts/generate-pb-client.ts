import * as fs from "fs";
import * as path from "path";

import { loadConfig } from "./config-loader";
import { setupStdbMock } from "./stdb-mock";

/**
 * Responsibility: Generate highly automated, metadata-driven PB metadata files.
 * Most logic has been migrated to src/utils/pb-client-utils.ts.
 */
async function main() {
  const config = loadConfig().pb_client;
  const OUTPUT_PATH = path.resolve("src/generated/pb_metadata.ts");
  const REDUCERS_DIR = path.resolve(config.reducers_dir);

  const capturedTables: any[] = [];
  const cleanupMock = setupStdbMock(capturedTables);

  // 1. Build PB index
  const pbRegistry = new Map<string, { importPath: string; schemaName: string }>();
  const scanProtoDir = (dir: string) => {
    if (!fs.existsSync(dir)) return;
    const files = fs.readdirSync(dir);
    for (const f of files) {
      const fullPath = path.join(dir, f);
      if (fs.statSync(fullPath).isDirectory()) {
        scanProtoDir(fullPath);
      } else if (f.endsWith("_pb.ts")) {
        const content = fs.readFileSync(fullPath, "utf8");
        const relativeToSrc = path.relative(path.resolve("src"), fullPath);
        const importPath = "@/" + relativeToSrc.split(path.sep).join("/").replace(".ts", "");

        const schemaMatches = content.matchAll(/export const (\w+Schema): GenMessage/g);
        for (const match of schemaMatches) {
          const schemaName = match[1];
          const typeName = schemaName.replace("Schema", "");
          pbRegistry.set(typeName, { importPath, schemaName });
        }
      }
    }
  };
  scanProtoDir(path.resolve("src/generated/flowcraft"));

  // 2. Load Reducer definitions
  const reducerFiles = fs.readdirSync(REDUCERS_DIR).filter((f) => f.endsWith(".ts"));
  const allReducers: Record<string, any> = {};
  for (const file of reducerFiles) {
    const mod = await import(path.join(REDUCERS_DIR, file));
    const key = Object.keys(mod).find((k) => k.toLowerCase().includes("reducer"));
    if (key) Object.assign(allReducers, mod[key]);
  }

  const TABLES_DIR = path.resolve("spacetime-module/src/tables");
  if (fs.existsSync(TABLES_DIR)) {
    const tableFiles = fs.readdirSync(TABLES_DIR).filter((f) => f.endsWith(".ts"));
    for (const file of tableFiles) {
      try {
        await import(path.join(TABLES_DIR, file));
      } catch (e) {
        console.warn(`Failed to import table file ${file}:`, e);
      }
    }
  }

  // 3. Build metadata and imports
  const importGroups = new Map<string, Set<string>>();
  const addImport = (typeName: string) => {
    const entry = pbRegistry.get(typeName);
    if (!entry) return null;
    if (!importGroups.has(entry.importPath)) importGroups.set(entry.importPath, new Set());
    importGroups.get(entry.importPath)!.add(entry.schemaName);
    return entry.schemaName;
  };

  const pbReducerEntries: string[] = [];
  for (const [rawName, def] of Object.entries(allReducers)) {
    const args = def.args;
    if (!args) continue;
    const camelName = toCamelCase(rawName);
    const fields: string[] = [];
    for (const [rawArgName, argTypeObj] of Object.entries(args)) {
      const argType = argTypeObj as any;

      let stName: string | undefined;
      if (argType?.__pb_schema) {
        stName = String(argType.__pb_schema);
      } else if (argType && typeof argType === "object" && "name" in argType) {
        stName = String(argType.name);
      }

      if (stName) {
        const matchName = pbRegistry.has(stName)
          ? stName
          : Array.from(pbRegistry.keys()).find((k) => stName.endsWith("_" + k));

        if (matchName) {
          const schemaName = addImport(matchName);
          if (schemaName) fields.push(`      ${toCamelCase(rawArgName)}: { schema: ${schemaName} }`);
        }
      }
    }
    if (fields.length > 0)
      pbReducerEntries.push(`  "${camelName}": {
${fields.join(",\n")} 
  }`);
  }

  const tableToProtoMetadata: string[] = [];
  for (const table of capturedTables) {
    for (const [colName, colType] of Object.entries(table.schema)) {
      const typeInfo = colType as any;
      if (typeInfo?.__st_name || typeInfo?.__pb_schema) {
        const stName = String(typeInfo.__pb_schema ?? typeInfo.__st_name);
        // Heuristic: exact match or stName ends with _TypeName (e.g. core_Node -> Node)
        const matchName = pbRegistry.has(stName)
          ? stName
          : Array.from(pbRegistry.keys()).find((k) => stName.endsWith("_" + k));

        if (matchName) {
          const schemaName = addImport(matchName);
          if (schemaName) {
            const accessorName = toCamelCase(String(table.name));
            tableToProtoMetadata.push(`  "${accessorName}": { schema: ${schemaName}, field: "${colName}" }`);
            break;
          }
        }
      }
    }
  }

  // 4. Generate file
  let importStatements = "";
  for (const [importPath, schemas] of importGroups.entries()) {
    importStatements += `import { ${Array.from(schemas).sort().join(", ")} } from "${importPath}";\n`;
  }

  const code = `/** AUTO-GENERATED - DO NOT EDIT **/ 
/* eslint-disable */
${importStatements}
import { type DbConnection } from "./spacetime";

/**
 * PB Override Manifest
 */
export const PB_REDUCERS_MAP = {
${pbReducerEntries.join(",\n")} 
} as const;

/**
 * Mapping between Tables and Protobuf Schemas
 */
export const TABLE_TO_PROTO = {
${tableToProtoMetadata.join(",\n")} 
} as const;

/**
 * Compile-time type safety assertion: ensures all mapped Reducers exist in the SDK
 */
type AssertReducersExist = keyof typeof PB_REDUCERS_MAP extends keyof DbConnection["reducers"]
  ? true
  : never;
export const _ASSERT_REDUCERS_SAFE: AssertReducersExist = true;
`;
  fs.mkdirSync(path.dirname(OUTPUT_PATH), { recursive: true });
  fs.writeFileSync(OUTPUT_PATH, code);
  cleanupMock();
  console.log("✅ PB Metadata generated at " + OUTPUT_PATH);
}

function toCamelCase(str: string): string {
  return str.replace(/([-_][a-z])/gi, ($1) => $1.toUpperCase().replace("-", "").replace("_", ""));
}

main().catch((err: unknown) => {
  console.error(err);
  process.exit(1);
});
