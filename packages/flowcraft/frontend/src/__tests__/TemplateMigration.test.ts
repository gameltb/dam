import { describe, it, expect } from "vitest";
import { NodeRegistry } from "../server/registry";
import "../server/templates"; // Trigger registration
import { create } from "@bufbuild/protobuf";
import { NodeTemplateSchema } from "../generated/flowcraft/v1/node_pb";

describe("NodeTemplate Migration", () => {
  it("should verify that current templates use the Protobuf schema", () => {
    const templates = NodeRegistry.getTemplates();
    templates.forEach((tpl) => {
      expect(tpl.templateId).toBeDefined();
      expect(tpl.displayName).toBeDefined();
      expect(tpl.menuPath).toBeDefined();
      expect(tpl.defaultState).toBeDefined();

      // Verify it's a valid Protobuf message by re-creating it
      const protoTemplate = create(NodeTemplateSchema, tpl);
      expect(protoTemplate.templateId).toBe(tpl.templateId);
    });
  });
});
