import { create } from "@bufbuild/protobuf";
import { AlgebraicType, BinaryWriter } from "spacetimedb";
import { describe, expect, it } from "vitest";

import { NodeKind, PresentationSchema } from "@/generated/flowcraft/v1/core/base_pb";
import { NodeDataSchema, NodeSchema, NodeTemplateSchema, RenderMode } from "@/generated/flowcraft/v1/core/node_pb";
import StdbNodeData from "@/generated/spacetime/node_data_type";
import StdbNodeTemplate from "@/generated/spacetime/node_template_type";
// Import SpacetimeDB Client Definitions
import StdbNode from "@/generated/spacetime/node_type";

import { pbToStdb, stdbToPb } from "../proto-stdb-bridge";

describe("proto-stdb-bridge (Client Schema)", () => {
  it("should convert a basic Node to SpacetimeDB object (Client Shape)", () => {
    const protoNode = create(NodeSchema, {
      isSelected: true,
      nodeId: "node-123",
      nodeKind: NodeKind.DYNAMIC,
      templateId: "template-abc",
    });

    const stdbType = StdbNode;
    const stdbObj = pbToStdb(NodeSchema, stdbType, protoNode) as any;

    expect(stdbObj.nodeId).toBe("node-123");
    expect(stdbObj.templateId).toBe("template-abc");
    // NodeKind is an enum (Sum). Proto=DYNAMIC(1).
    // In SpacetimeDB Client with { tag, value } format:
    expect(stdbObj.nodeKind).toEqual({ tag: "NODE_KIND_DYNAMIC", value: {} });
    expect(stdbObj.isSelected).toBe(true);
  });

  it("should convert NodeTemplate to SpacetimeDB object", () => {
    const protoTemplate = create(NodeTemplateSchema, {
      defaultHeight: 150,
      defaultWidth: 200,
      displayName: "My Template",
      menuPath: ["AI", "Generators"],
      templateId: "tpl-1",
      widgetsSchema: {
        properties: {
          foo: { type: "string" },
        },
        type: "object",
      },
    });

    const stdbType = StdbNodeTemplate;
    const stdbObj = pbToStdb(NodeTemplateSchema, stdbType, protoTemplate) as any;

    expect(stdbObj.templateId).toBe("tpl-1");
    expect(stdbObj.displayName).toBe("My Template");
    expect(stdbObj.menuPath).toEqual(["AI", "Generators"]);
    expect(stdbObj.defaultWidth).toBe(200);
    expect(stdbObj.defaultHeight).toBe(150);
    // widgetsSchema is google.protobuf.Struct which is mapped to string (JSON) in STDB schema script
    expect(typeof stdbObj.widgetsSchema).toBe("string");
    expect(JSON.parse(stdbObj.widgetsSchema).properties.foo.type).toBe("string");

    const writer = new BinaryWriter(1024);
    const algType = (StdbNodeTemplate as any).algebraicType || StdbNodeTemplate;

    expect(() => {
      AlgebraicType.serializeValue(writer, algType, stdbObj);
    }).not.toThrow();
  });

  it("should convert nested Presentation message", () => {
    const protoNode = create(NodeSchema, {
      presentation: create(PresentationSchema, {
        height: 400,
        position: { x: 100, y: 200 },
        width: 300,
      }),
    });

    const stdbType = StdbNode;
    const stdbObj = pbToStdb(NodeSchema, stdbType, protoNode) as any;

    expect(stdbObj.presentation).toBeDefined();
    expect(stdbObj.presentation.position).toEqual({ x: 100, y: 200 });
    expect(stdbObj.presentation.width).toBe(300);
    expect(stdbObj.presentation.height).toBe(400);
  });

  it("should round-trip (PB -> STDB -> PB)", () => {
    const originalProto = create(NodeSchema, {
      isSelected: false,
      nodeId: "node-roundtrip",
      nodeKind: NodeKind.GROUP,
      presentation: create(PresentationSchema, {
        position: { x: 10, y: 20 },
      }),
    });

    const stdbType = StdbNode;
    const stdbObj = pbToStdb(NodeSchema, stdbType, originalProto);
    const restoredProto = stdbToPb(NodeSchema, stdbType, stdbObj);

    expect(restoredProto.nodeId).toBe(originalProto.nodeId);
    expect(restoredProto.nodeKind).toBe(originalProto.nodeKind);
    expect(restoredProto.presentation?.position?.x).toBe(originalProto.presentation?.position?.x);
    expect(restoredProto.presentation?.position?.y).toBe(originalProto.presentation?.position?.y);
    expect(restoredProto.isSelected).toBe(originalProto.isSelected);
  });

  it("should be serializable by SpacetimeDB SDK", () => {
    const protoNode = create(NodeSchema, {
      nodeId: "node-serializable",
      nodeKind: NodeKind.DYNAMIC,
      presentation: create(PresentationSchema, { width: 100 }),
    });

    const stdbType = StdbNode;
    const stdbObj = pbToStdb(NodeSchema, stdbType, protoNode);

    const writer = new BinaryWriter(1024);

    // StdbNode from generated/spacetime/node_type.ts export __t.object(...)
    // which has .algebraicType property in recent STDB versions?
    // Or it IS a TypeDef.
    const algType = (StdbNode as any).algebraicType || StdbNode;

    expect(() => {
      AlgebraicType.serializeValue(writer, algType, stdbObj);
    }).not.toThrow();
  });

  it("should handle Enum and Repeated Enum (List)", () => {
    const protoData = create(NodeDataSchema, {
      activeMode: RenderMode.MODE_CHAT,
      availableModes: [RenderMode.MODE_MEDIA, RenderMode.MODE_WIDGETS],
      displayName: "Test Enums",
      metadata: {
        key1: "value1",
        key2: "value2",
      },
    });

    const stdbType = StdbNodeData;
    const stdbObj = pbToStdb(NodeDataSchema, stdbType, protoData) as any;

    expect(stdbObj.displayName).toBe("Test Enums");

    // Check metadata map conversion to STDB array
    expect(Array.isArray(stdbObj.metadata)).toBe(true);
    expect(stdbObj.metadata).toHaveLength(2);
    expect(stdbObj.metadata).toContainEqual({ key: "key1", value: "value1" });
    expect(stdbObj.metadata).toContainEqual({ key: "key2", value: "value2" });

    // Check single Enum (wrapped in Option in NodeData schema, but pbToStdb returns value directly,
    // BUT the Enum itself is a SumType { tag, value })
    expect(stdbObj.activeMode).toEqual({ tag: "MODE_CHAT", value: {} });

    // Check Repeated Enum
    expect(Array.isArray(stdbObj.availableModes)).toBe(true);
    expect(stdbObj.availableModes).toHaveLength(2);
    expect(stdbObj.availableModes[0]).toEqual({ tag: "MODE_MEDIA", value: {} });
    expect(stdbObj.availableModes[1]).toEqual({
      tag: "MODE_WIDGETS",
      value: {},
    });

    // Round trip
    const restored = stdbToPb(NodeDataSchema, stdbType, stdbObj);
    expect(restored.activeMode).toBe(RenderMode.MODE_CHAT);
    expect(restored.availableModes).toEqual([RenderMode.MODE_MEDIA, RenderMode.MODE_WIDGETS]);
    expect(restored.metadata).toEqual({
      key1: "value1",
      key2: "value2",
    });

    // Serialize
    const writer = new BinaryWriter(1024);
    const algType = (StdbNodeData as any).algebraicType || StdbNodeData;
    expect(() => {
      AlgebraicType.serializeValue(writer, algType, stdbObj);
    }).not.toThrow();
  });
});
