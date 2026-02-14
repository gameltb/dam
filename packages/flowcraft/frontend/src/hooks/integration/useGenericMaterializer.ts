import { useEffect } from "react";

import { commit } from "@/store/orchestrator";

/**
 * useGenericMaterializer
 */
export const useGenericMaterializer = () => {
  useEffect(() => {
    // Materialization logic
  }, []);

  const materialize = (_data: any) => {
    commit(
      (_draft) => {
        // Materialize logic
      },
      { description: "Materialize data" },
    );
  };

  return { materialize };
};
