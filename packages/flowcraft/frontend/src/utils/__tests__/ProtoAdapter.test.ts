import { create } from "@bufbuild/protobuf";
import { describe, expect, it } from "vitest";

import {
  NodeKind,
  PresentationSchema,
} from "../../generated/flowcraft/v1/core/base_pb";
import { NodeSchema } from "../../generated/flowcraft/v1/core/node_pb";
import { fromProtoNode } from "../protoAdapter";

/**
 * @file ProtoAdapter.test.ts
 * PROBLEM: Nodes were invisible in the MiniMap.
 * CAUSE: ProtoAdapter forced missing width/height to 0, which overrode React Flow's auto-measurement.
 * REQUIREMENT: Ensure measured dimensions are only set if positive values exist in the protocol.
 */
describe("ProtoAdapter", () => {
  it("should NOT set measured dimensions when proto width/height are 0", () => {
    const protoNode = create(NodeSchema, {
      isSelected: false,
      nodeId: "node-1",
      nodeKind: NodeKind.DYNAMIC,
      presentation: create(PresentationSchema, {
        height: 0,
        parentId: "",
        position: { x: 10, y: 10 },
        width: 0,
      }),
    });

    const appNode = fromProtoNode(protoNode);

    // Crucial for MiniMap: measured should be undefined so React Flow can measure it
    expect(appNode.measured).toBeUndefined();
  });

  it("should set measured dimensions when proto width/height are positive", () => {
    const protoNode = create(NodeSchema, {
      isSelected: false,
      nodeId: "node-1",
      nodeKind: NodeKind.DYNAMIC,
      presentation: create(PresentationSchema, {
        height: 100,
        parentId: "",
        position: { x: 10, y: 10 },
        width: 150,
      }),
    });

    const appNode = fromProtoNode(protoNode);

    expect(appNode.measured).toEqual({ height: 100, width: 150 });
  });
});
