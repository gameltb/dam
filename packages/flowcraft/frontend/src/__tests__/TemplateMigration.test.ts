import { create } from "@bufbuild/protobuf";
import { describe, expect, it } from "vitest";

import "../server/templates"; // Trigger registration
import { NodeTemplateSchema } from "@/generated/flowcraft/v1/core/node_pb";
import { NodeRegistry } from "../server/services/NodeRegistry";

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
