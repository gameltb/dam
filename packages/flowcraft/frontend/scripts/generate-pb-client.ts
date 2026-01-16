import * as fs from "fs";
import * as path from "path";

import { loadConfig } from "./config-loader";
import { setupStdbMock } from "./stdb-mock";

/**
 * 职责：生成高度自动化的、元数据驱动的 PB 元数据文件。
 * 大部分逻辑已迁移至 src/utils/pb-client-utils.ts。
 */
async function main() {
  const config = loadConfig().pb_client;
  const OUTPUT_PATH = path.resolve("src/generated/pb_metadata.ts");
  const REDUCERS_DIR = path.resolve(config.reducers_dir);
  const TABLES_DIR = path.resolve("spacetime-module/src/tables");

  const capturedTables: any[] = [];
  const cleanupMock = setupStdbMock(capturedTables);

  // 1. 构建 PB 索引
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

  // 2. 加载 Reducer 定义
  const reducerFiles = fs.readdirSync(REDUCERS_DIR).filter((f) => f.endsWith(".ts"));
  const allReducers: Record<string, any> = {};
  for (const file of reducerFiles) {
    const mod = await import(path.join(REDUCERS_DIR, file));
    const key = Object.keys(mod).find((k) => k.toLowerCase().includes("reducer"));
    if (key) Object.assign(allReducers, mod[key]);
  }

  if (fs.existsSync(TABLES_DIR)) {
    const tableFiles = fs.readdirSync(TABLES_DIR).filter((f) => f.endsWith(".ts"));
    for (const file of tableFiles) {
      try {
        await import(path.join(TABLES_DIR, file));
      } catch (e) {
        // Ignore files that fail to load in Node context due to missing browser/stdb deps
      }
    }
  }

  // 3. 构建元数据和导入
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
      if (argType && typeof argType === "object" && "name" in argType) {
        const schemaName = addImport(argType.name);
        if (schemaName) fields.push(`      ${toCamelCase(rawArgName)}: { schema: ${schemaName} }`);
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
      if (typeInfo?.__st_name) {
        const stName = String(typeInfo.__st_name);
        const matchName = pbRegistry.has(stName)
          ? stName
          : Array.from(pbRegistry.keys()).find((k) => k.endsWith("_" + stName));
        if (matchName) {
          const schemaName = addImport(matchName);
          if (schemaName) {
            // 使用 toCamelCase(table.name) 作为键，匹配前端 tables 对象的属性名
            const accessorName = toCamelCase(String(table.name));
            tableToProtoMetadata.push(`  "${accessorName}": { schema: ${schemaName}, field: "${colName}" }`);
            break;
          }
        }
      }
    }
  }

  // 4. 生成文件
  let importStatements = "";
  for (const [importPath, schemas] of importGroups.entries()) {
    importStatements += `import { ${Array.from(schemas).sort().join(", ")} } from "${importPath}";\n`;
  }

  const code = `/** AUTO-GENERATED - DO NOT EDIT **/ 
/* eslint-disable */
${importStatements}

/**
 * PB 覆盖清单
 */
export const PB_REDUCERS_MAP = {
${pbReducerEntries.join(",\n")} 
} as const;

/**
 * 表与 Protobuf Schema 的映射
 */
export const TABLE_TO_PROTO: Record<string, { schema: any, field: string }> = {
${tableToProtoMetadata.join(",\n")} 
};
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
