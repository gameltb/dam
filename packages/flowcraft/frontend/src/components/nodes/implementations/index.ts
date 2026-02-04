import { AiGenNodeStateSchema } from "@/generated/flowcraft/v1/nodes/ai_gen_node_pb";
// State Schemas from generated code
import { ChatNodeStateSchema } from "@/generated/flowcraft/v1/nodes/chat_node_pb";
import {
  AcousticNodeStateSchema,
  DocumentNodeStateSchema,
  VisualNodeStateSchema,
} from "@/generated/flowcraft/v1/nodes/media_node_pb";
import { SubgraphNodeStateSchema } from "@/generated/flowcraft/v1/nodes/subgraph_node_pb";

import { registerNode } from "../core/NodeRegistry";
import { AiGenNodeImplementation } from "./AiGenNode";
// Specialized components
import { ChatNodeImplementation } from "./ChatNode";
import { MediaNodeImplementation } from "./MediaNode";
import { SubgraphNodeImplementation } from "./SubgraphNode";

/**
 * Initialize all manual node registrations based on Protobuf Schemas.
 * This establishes a 1:1 link between a Proto Type and a React Component.
 */
export const initNodeRegistry = () => {
  // 1. Chat
  registerNode({
    component: ChatNodeImplementation as any,
    constraints: { minHeight: 400, minWidth: 350 },
    schema: ChatNodeStateSchema,
  });

  // 2. Media Types (Multiple keys can map to specialized logic)
  const mediaConstraints = { minHeight: 150, minWidth: 200 };
  registerNode({
    component: MediaNodeImplementation as any,
    constraints: mediaConstraints,
    schema: VisualNodeStateSchema,
  });
  registerNode({
    component: MediaNodeImplementation as any,
    constraints: mediaConstraints,
    schema: DocumentNodeStateSchema,
  });
  registerNode({
    component: MediaNodeImplementation as any,
    constraints: mediaConstraints,
    schema: AcousticNodeStateSchema,
  });

  // 3. AI Gen
  registerNode({ component: AiGenNodeImplementation as any, schema: AiGenNodeStateSchema });

  // 4. Subgraph
  registerNode({ component: SubgraphNodeImplementation as any, schema: SubgraphNodeStateSchema });

  console.log("[NodeRegistry] Manual registrations via Schema objects complete.");
};
