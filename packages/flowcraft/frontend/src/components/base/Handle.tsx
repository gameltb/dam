// src/components/base/Handle.tsx

import { type HandleProps, Handle as ReactFlowHandle } from "@xyflow/react";
import React from "react";

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
