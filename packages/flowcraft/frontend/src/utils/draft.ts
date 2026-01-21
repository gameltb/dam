import { type DescMessage } from "@bufbuild/protobuf";

import { createSchemaDraft } from "./schemaProxy";

/**
 * Rust-style Result type for safe error handling.
 */
export type Result<T, E = string> =
  | { readonly error: E; readonly ok: false }
  | { readonly ok: true; readonly value: T };

export const Ok = <T>(value: T): Result<T, never> => ({ ok: true, value });
export const Err = <E>(error: E): Result<never, E> => ({ error, ok: false });

/**
 * Rust-style Option type.
 */
export type Option<T> = { readonly some: false } | { readonly some: true; readonly value: T };

export const Some = <T>(value: T): Option<T> => ({ some: true, value });
export const None = (): Option<never> => ({ some: false });

/**
 * Type Gymnastics: Recursively makes a type writable and handles PB message shapes.
 */
export type Draftable<T> = T extends object
  ? T extends Uint8Array
    ? T
    : {
        -readonly [K in keyof T]: Draftable<T[K]>;
      }
  : T;

/**
 * Special handling for Node messages to bridge React Flow and Protobuf.
 * Aligns 'data' property with PB 'state'.
 */
export type NodeDraft<TMessage extends object> = Draftable<TMessage> & {
  /** Alias for state to satisfy React Flow requirements */
  data: TMessage extends { state?: infer S } ? Draftable<S> : any;
};

/**
 * Generic Draft container.
 */
export class Draft<T extends object> {
  public readonly proxy: Draftable<T>;

  constructor(
    target: T,
    schema: DescMessage,
    onCommit: (path: string, value: unknown) => void,
    options: { pathMapper?: (prop: string) => null | string } = {},
  ) {
    this.proxy = createSchemaDraft(target, schema, onCommit, options) as Draftable<T>;
  }
}

/**
 * Node-specific mapper to bridge UI names to Protocol names.
 */
export const NODE_PATH_MAPPER = (prop: string): null | string => {
  const map: Record<string, string> = {
    data: "state",
    height: "presentation.height",
    parentId: "presentation.parent_id",
    position: "presentation.position",
    selected: "presentation.is_selected",
    width: "presentation.width",
  };
  return map[prop] || null;
};

/**
 * Helper to create a Node draft with standard mappings.
 */
export function createNodeDraft<T extends object>(
  nodeId: string,
  target: T,
  schema: DescMessage,
  commitFn: (path: string, value: any) => void,
): Result<Draftable<T>> {
  if (!target) return Err(`Node ${nodeId} not found`);

  const draft = new Draft(target, schema, commitFn, { pathMapper: NODE_PATH_MAPPER });
  return Ok(draft.proxy);
}
