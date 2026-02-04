import React, { createContext, useContext } from "react";

import { type BindingBackend } from "@/utils/lens-types";

const BindingContext = createContext<null | Record<string, BindingBackend<any>>>(null);

/**
 * BindingProvider
 * Allows overriding the synchronization backend for a specific category in a local tree.
 */
export const BindingProvider: React.FC<{
  backends: Record<string, BindingBackend<any>>;
  children: React.ReactNode;
}> = ({ backends, children }) => {
  const parentBackends = useContext(BindingContext) || {};
  const mergedBackends = { ...parentBackends, ...backends };

  return <BindingContext.Provider value={mergedBackends}>{children}</BindingContext.Provider>;
};

export const useBindingBackends = () => useContext(BindingContext);
