import React from "react";

import { useNavigationStore } from "@/store/ui/navigationStore";

/**
 * withScopeLens
 * HOC that injects the current activeScopeId into a component.
 */
export function withScopeLens<P extends object>(Component: React.ComponentType<P & { activeScopeId: null | string }>) {
  return (props: P) => {
    const activeScopeId = useNavigationStore((s) => s.activeScopeId);
    return <Component {...props} activeScopeId={activeScopeId} />;
  };
}
