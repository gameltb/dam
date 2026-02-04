import { useCallback, useEffect, useRef } from "react";
import { useSpacetimeDB } from "spacetimedb/react";

import { DbConnection } from "@/generated/spacetime";
import { useFlowStore } from "@/store/flowStore";
import { wrapReducers } from "@/utils/pb-client";

/**
 * useSpacetimeConnection
 *
 * Responsible ONLY for connection management and identity.
 * Subscription and data-sync are delegated to specialized hooks.
 */
export const useSpacetimeConnection = () => {
  const stdb = useSpacetimeDB();
  const { isActive } = stdb;
  const connInitializedRef = useRef<boolean>(false);

  const getConnection = useCallback(() => stdb.getConnection<DbConnection>(), [stdb]);

  useEffect(() => {
    const conn = getConnection();
    if (conn && isActive && !connInitializedRef.current) {
      const pbConn = wrapReducers(conn);
      useFlowStore.setState({ spacetimeConn: pbConn });
      connInitializedRef.current = true;

      const sessionTaskId = `user-session-${crypto.randomUUID()}`;
      pbConn.reducers.assignCurrentTask({ taskId: sessionTaskId });

      console.log("[Spacetime] Connection active. Ready for subscriptions.");
    } else if (!isActive) {
      connInitializedRef.current = false;
    }
  }, [getConnection, isActive]);

  return { getConnection, isActive };
};
