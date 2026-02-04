import { create } from "@bufbuild/protobuf";
import { v4 as uuidv4 } from "uuid";

import { NodeDataSchema, NodeTemplateSchema, RenderMode } from "@/generated/flowcraft/v1/core/node_pb";
import { SubgraphNodeStateSchema } from "@/generated/flowcraft/v1/nodes/subgraph_node_pb";

import { NodeRegistry } from "../services/NodeRegistry";
import { SubgraphProxyInstance } from "./SubgraphProxyInstance";

NodeRegistry.register({
  // Use subgraph proxy instance
  createInstance: (nodeId: string) => new SubgraphProxyInstance(uuidv4(), nodeId),
  schema: SubgraphNodeStateSchema,
  template: create(NodeTemplateSchema, {
    defaultState: create(NodeDataSchema, {
      activeMode: RenderMode.MODE_MEDIA,
      availableModes: [RenderMode.MODE_MEDIA, RenderMode.MODE_WIDGETS],
      displayName: "Imported Session",
      extension: {
        case: "subgraph",
        value: create(SubgraphNodeStateSchema, {
          originalSource: "import",
          subgraphId: "",
        }),
      },
    }),
    displayName: "Subgraph Proxy",
    menuPath: ["General"],
  }),
});
