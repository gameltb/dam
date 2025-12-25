import { flowcraft } from "../generated/flowcraft";
import type { AppNode, DynamicNodeData, WidgetDef, MediaDef } from "../types";
import { RenderMode } from "../types";
import type { Edge } from "@xyflow/react";

/**
 * Adapter to convert between Protobuf generated types and internal React Flow types.
 */
export class ProtoAdapter {
  // --- To Proto ---

  static toProtoNode(node: AppNode): flowcraft.v1.INode {
    const isDynamic = node.type === "dynamic";
    const data = isDynamic ? (node.data as DynamicNodeData) : null;

    return {
      id: node.id,
      type: node.type,
      position: node.position,
      width: node.measured?.width || (node.style?.width as number) || 0,
      height: node.measured?.height || (node.style?.height as number) || 0,
      selected: !!node.selected,
      parentId: node.parentId || "",
      data: data ? ProtoAdapter.toProtoNodeData(data) : null,
    };
  }

  static toProtoNodeData(data: DynamicNodeData): flowcraft.v1.INodeData {
    return {
      label: data.label,
      availableModes: data.modes as unknown as flowcraft.v1.RenderMode[],
      activeMode:
        (data.activeMode as unknown as flowcraft.v1.RenderMode) ||
        RenderMode.MODE_UNSPECIFIED,
      media: data.media ? ProtoAdapter.toProtoMedia(data.media) : null,
      widgets: data.widgets ? data.widgets.map(ProtoAdapter.toProtoWidget) : [],
      inputPorts: (data.inputPorts as flowcraft.v1.IPort[]) || [],
      outputPorts: (data.outputPorts as flowcraft.v1.IPort[]) || [],
      metadata: {},
    };
  }

  static toProtoMedia(
    media: MediaDef & { aspectRatio?: number },
  ): flowcraft.v1.IMediaContent {
    return {
      type: media.type as unknown as flowcraft.v1.MediaType,
      url: media.url || "",
      content: media.content || "",
      galleryUrls: media.galleryUrls || [],
      aspectRatio: media.aspectRatio || 0,
    };
  }

  static toProtoWidget(widget: WidgetDef): flowcraft.v1.IWidget {
    return {
      id: widget.id,
      type: widget.type as unknown as flowcraft.v1.WidgetType,
      label: widget.label,
      valueJson: JSON.stringify(widget.value),
      options:
        widget.options?.map((o) => ({
          label: o.label,
          value: String(o.value),
          description: "",
        })) || [],
      config: {
        placeholder: "",
        min: (widget.config?.min as number) || 0,
        max: (widget.config?.max as number) || 0,
        step: (widget.config?.step as number) || 0,
        dynamicOptions: !!widget.config?.dynamicOptions,
        actionTarget: "",
      },
      inputPortId: widget.inputPortId || "",
    };
  }

  // --- From Proto ---

  static fromProtoGraph(protoGraph: flowcraft.v1.IGraphSnapshot): {
    nodes: AppNode[];
    edges: Edge[];
  } {
    if (!protoGraph.nodes) return { nodes: [], edges: [] };

    const nodes: AppNode[] = protoGraph.nodes.map((n) => {
      return {
        id: n.id!,
        position: { x: n.position?.x || 0, y: n.position?.y || 0 },
        data: {} as DynamicNodeData,
        type: "dynamic",
      } as AppNode;
    });

    return { nodes, edges: [] };
  }
}
