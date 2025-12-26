import { describe, it, expect } from "vitest";
import { ProtoAdapter } from "../protoAdapter";
import { flowcraft } from "../../generated/flowcraft";

/**
 * PROBLEM: Nodes were invisible in the MiniMap.
 * CAUSE: ProtoAdapter forced missing width/height to 0, which overrode React Flow's auto-measurement.
 * REQUIREMENT: Ensure measured dimensions are only set if positive values exist in the protocol.
 */
describe("ProtoAdapter", () => {
  it("should NOT set measured dimensions when proto width/height are 0", () => {
    const protoNode: flowcraft.v1.INode = {
      id: "node-1",
      position: { x: 10, y: 10 },
      width: 0,
      height: 0,
    };

    const appNode = ProtoAdapter.fromProtoNode(protoNode);

    // Crucial for MiniMap: measured should be undefined so React Flow can measure it
    expect(appNode.measured).toBeUndefined();
  });

  it("should set measured dimensions when proto width/height are positive", () => {
    const protoNode: flowcraft.v1.INode = {
      id: "node-1",
      position: { x: 10, y: 10 },
      width: 150,
      height: 100,
    };

    const appNode = ProtoAdapter.fromProtoNode(protoNode);

    expect(appNode.measured).toEqual({ width: 150, height: 100 });
  });
});
