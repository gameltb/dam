import React, { createContext, useContext } from "react";

const NodeContext = createContext<null | { nodeId: string }>(null);

/**
 * NodeProvider
 *
 * Injects node identity into the component sub-tree.
 * Allows child components to access node data without prop-drilling nodeId.
 */
export const NodeProvider: React.FC<{ children: React.ReactNode; nodeId: string }> = ({ children, nodeId }) => {
  return <NodeContext.Provider value={{ nodeId }}>{children}</NodeContext.Provider>;
};

/**
 * useNodeId
 *
 * Safe hook to get the current node ID from context.
 */
export const useNodeId = () => {
  const ctx = useContext(NodeContext);
  if (!ctx) {
    throw new Error("useNodeId must be used within a NodeProvider");
  }
  return ctx.nodeId;
};
