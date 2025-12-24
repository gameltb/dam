// src/components/base/Handle.tsx

import React from "react";
import { Handle as ReactFlowHandle, type HandleProps } from "@xyflow/react";

export const Handle: React.FC<HandleProps> = (props) => {
  return (
    <ReactFlowHandle
      {...props}
      style={{
        zIndex: 10,
        ...props.style,
      }}
    />
  );
};
