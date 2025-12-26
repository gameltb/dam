import { flowcraft } from "../generated/flowcraft";
import type { AppNode, DynamicNodeData, WidgetDef, MediaDef } from "../types";
import { RenderMode, MediaType, WidgetType } from "../types";
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
    const nodes: AppNode[] = (protoGraph.nodes || []).map(
      ProtoAdapter.fromProtoNode,
    );
    const edges: Edge[] = (protoGraph.edges || []).map((e) => ({
      id: e.id!,
      source: e.source!,
      target: e.target!,
      sourceHandle: e.sourceHandle || undefined,
      targetHandle: e.targetHandle || undefined,
      data: e.metadata || {},
    }));

    return { nodes, edges };
  }

  static fromProtoNode(n: flowcraft.v1.INode): AppNode {
    const rawType = n.type || "dynamic";
    const isStandardSpecial =
      rawType === "groupNode" || rawType === "processing";
    const reactFlowType = isStandardSpecial ? rawType : "dynamic";

    const node: AppNode = {
      id: n.id!,
      type: reactFlowType,
      position: { x: n.position?.x || 0, y: n.position?.y || 0 },
      selected: !!n.selected,
      parentId: n.parentId || undefined,
      data: n.data
        ? ProtoAdapter.fromProtoNodeData(n.data)
        : ({} as DynamicNodeData),
    } as AppNode;

    if (!isStandardSpecial && node.type === "dynamic") {
      (node.data as DynamicNodeData).typeId = rawType;
    }

    if (n.width && n.height) {
      node.measured = { width: n.width, height: n.height };
    }

    return node;
  }

  static fromProtoNodeData(data: flowcraft.v1.INodeData): DynamicNodeData {
    return {
      label: data.label || "",
      modes: (data.availableModes || []) as unknown as RenderMode[],
      activeMode: (data.activeMode ||
        RenderMode.MODE_WIDGETS) as unknown as RenderMode,
      media: data.media ? ProtoAdapter.fromProtoMedia(data.media) : undefined,
      widgets: (data.widgets || []).map(ProtoAdapter.fromProtoWidget),
      inputPorts: (data.inputPorts || []) as flowcraft.v1.IPort[],
      outputPorts: (data.outputPorts || []) as flowcraft.v1.IPort[],
      // Handlers will be attached during hydration
      onChange: () => {},
    };
  }

  static fromProtoMedia(media: flowcraft.v1.IMediaContent): MediaDef {
    return {
      type: media.type as unknown as MediaType,
      url: media.url || undefined,
      content: media.content || undefined,
      galleryUrls: media.galleryUrls || [],
    };
  }

  static fromProtoWidget(widget: flowcraft.v1.IWidget): WidgetDef {
    let value: unknown = undefined;
    try {
      value = widget.valueJson ? JSON.parse(widget.valueJson) : undefined;
    } catch (e) {
      console.error("Failed to parse widget valueJson", e);
    }

    return {
      id: widget.id!,
      type: widget.type as unknown as WidgetType,
      label: widget.label || "",
      value: value,
      options: (widget.options || []).map((o) => ({
        label: o.label || "",
        value: o.value || "",
      })),
      config: {
        min: widget.config?.min ?? undefined,
        max: widget.config?.max ?? undefined,
        step: widget.config?.step ?? undefined,
        dynamicOptions: !!widget.config?.dynamicOptions,
      },
      inputPortId: widget.inputPortId || undefined,
    };
  }
}