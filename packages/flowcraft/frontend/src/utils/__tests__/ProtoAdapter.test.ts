import { describe, it, expect } from "vitest";
import { fromProtoNode } from "../protoAdapter";
import { NodeSchema } from "../../generated/flowcraft/v1/node_pb";
import {
  PresentationSchema,
  NodeKind,
} from "../../generated/flowcraft/v1/base_pb";
import { create } from "@bufbuild/protobuf";

/**
 * @file ProtoAdapter.test.ts
 * PROBLEM: Nodes were invisible in the MiniMap.
 * CAUSE: ProtoAdapter forced missing width/height to 0, which overrode React Flow's auto-measurement.
 * REQUIREMENT: Ensure measured dimensions are only set if positive values exist in the protocol.
 */
describe("ProtoAdapter", () => {
  it("should NOT set measured dimensions when proto width/height are 0", () => {
    const protoNode = create(NodeSchema, {
      nodeId: "node-1",
      presentation: create(PresentationSchema, {
        position: { x: 10, y: 10 },
        width: 0,
        height: 0,
        parentId: "",
      }),
      nodeKind: NodeKind.DYNAMIC,
      isSelected: false,
    });

    const appNode = fromProtoNode(protoNode);

    // Crucial for MiniMap: measured should be undefined so React Flow can measure it
    expect(appNode.measured).toBeUndefined();
  });

  it("should set measured dimensions when proto width/height are positive", () => {
    const protoNode = create(NodeSchema, {
      nodeId: "node-1",
      presentation: create(PresentationSchema, {
        position: { x: 10, y: 10 },
        width: 150,
        height: 100,
        parentId: "",
      }),
      nodeKind: NodeKind.DYNAMIC,
      isSelected: false,
    });

    const appNode = fromProtoNode(protoNode);

    expect(appNode.measured).toEqual({ width: 150, height: 100 });
  });
});
