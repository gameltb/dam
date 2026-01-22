import { PB_REDUCERS_MAP, TABLE_TO_PROTO } from "@/generated/pb_metadata";
import { type DbConnection, tables } from "@/generated/spacetime";
import { NodeKernel } from "@/kernel/NodeKernel";

import { convertStdbToPbInternal, type PbReducersProjection, wrapReducersInternal } from "./pb-client-utils";

// 保持兼容性别名
export type PbClient = PbConnection;

/**
 * 包装后的连接器类型
 */
export type PbConnection = DbConnection & {
  kernel: NodeKernel;
  pbreducers: ProjectedReducers;
};

/**
 * 核心类型：通过投影自动生成 PB 增强版的 Reducers 签名
 */
type ProjectedReducers = PbReducersProjection<DbConnection["reducers"], typeof PB_REDUCERS_MAP>;

/**
 * 将 STDB Row 转换为标准的 PB 对象
 * 自动使用全局导出的 tables 映射，无需手动传递
 */
export function convertStdbToPb(tableName: string, row: any): any {
  return convertStdbToPbInternal(tableName, row, tables, TABLE_TO_PROTO);
}

/**
 * 核心包装函数：将 DbConnection 升级为支持 PB 自动序列化的版本
 */
export function wrapReducers(conn: DbConnection): PbConnection {
  const wrapped = wrapReducersInternal(conn, PB_REDUCERS_MAP);
  wrapped.kernel = new NodeKernel(wrapped);
  return wrapped as PbConnection;
}
