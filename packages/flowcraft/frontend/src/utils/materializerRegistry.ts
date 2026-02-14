import { type PbConnection } from "./pb-client";

export interface Materializer {
  name: string;
  // Now setup receives activeScopeId and returns a cleanup function
  setup: (conn: PbConnection, activeScopeId: null | string) => (() => void) | void;
}

const registry: Materializer[] = [];

export const registerMaterializer = (m: Materializer) => {
  registry.push(m);
};

export const getMaterializers = () => registry;
