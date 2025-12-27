import { flowcraft_proto } from "../generated/flowcraft_proto";
import type {
  AppNode,
  DynamicNodeData,
  WidgetDef,
  MediaDef,
  MediaType,
  WidgetType,
  RenderMode,
} from "../types";
import type { Edge } from "@xyflow/react";

/**
 * Adapter functions to convert between Protobuf generated types and internal React Flow types.
 */
const RenderModeVal = flowcraft_proto.v1.RenderMode;

// --- To Proto ---

export function toProtoNode(node: AppNode): flowcraft_proto.v1.INode {
  const isDynamic = node.type === "dynamic";
  const data = isDynamic ? node.data : null;

  return {
    id: node.id,
    type: node.type,
    position: node.position,
    width:
      node.measured?.width ?? (node.style?.width as number | undefined) ?? 0,
    height:
      node.measured?.height ?? (node.style?.height as number | undefined) ?? 0,
    selected: !!node.selected,
    parentId: node.parentId ?? "",
    data: data ? toProtoNodeData(data) : null,
  };
}

export function toProtoNodeData(
  data: DynamicNodeData,
): flowcraft_proto.v1.INodeData {
  return {
    label: data.label,
    availableModes: data.modes as unknown as flowcraft_proto.v1.RenderMode[],
    activeMode:
      (data.activeMode as unknown as flowcraft_proto.v1.RenderMode),
    media: data.media ? toProtoMedia(data.media) : null,
    widgets: data.widgets ? data.widgets.map(toProtoWidget) : [],
    inputPorts: data.inputPorts ?? [],
    outputPorts: data.outputPorts ?? [],
    metadata: {},
  };
}

export function toProtoMedia(
  media: MediaDef & { aspectRatio?: number },
): flowcraft_proto.v1.IMediaContent {
  return {
    type: media.type as unknown as flowcraft_proto.v1.MediaType,
    url: media.url ?? "",
    content: media.content ?? "",
    galleryUrls: media.galleryUrls ?? [],
    aspectRatio: media.aspectRatio ?? 0,
  };
}

export function toProtoWidget(widget: WidgetDef): flowcraft_proto.v1.IWidget {
  return {
    id: widget.id,
    type: widget.type as unknown as flowcraft_proto.v1.WidgetType,
    label: widget.label,
    valueJson: JSON.stringify(widget.value),
    options:
      widget.options?.map((o) => ({
        label: o.label,
        value: String(o.value),
        description: "",
      })) ?? [],
    config: {
      placeholder: "",
      min: (widget.config?.min as number | undefined) ?? 0,
      max: (widget.config?.max as number | undefined) ?? 0,
      step: (widget.config?.step as number | undefined) ?? 0,
      dynamicOptions: !!widget.config?.dynamicOptions,
      actionTarget: "",
    },
    inputPortId: widget.inputPortId ?? "",
  };
}

// --- From Proto ---

export function fromProtoGraph(protoGraph: flowcraft_proto.v1.IGraphSnapshot): {
  nodes: AppNode[];
  edges: Edge[];
} {
  const nodes: AppNode[] = (protoGraph.nodes ?? []).map((n) =>
    fromProtoNode(n),
  );
  const edges: Edge[] = (protoGraph.edges ?? []).map((e) => ({
    id: e.id ?? "",
    source: e.source ?? "",
    target: e.target ?? "",
    sourceHandle: e.sourceHandle ?? undefined,
    targetHandle: e.targetHandle ?? undefined,
    data: (e.metadata as Record<string, unknown> | undefined) ?? {},
  }));

  return { nodes, edges };
}

export function fromProtoNode(n: flowcraft_proto.v1.INode): AppNode {
  const rawType = n.type ?? "dynamic";
  const isStandardSpecial = rawType === "groupNode" || rawType === "processing";
  const reactFlowType = isStandardSpecial ? rawType : "dynamic";

  const node: AppNode = {
    id: n.id ?? "",
    type: reactFlowType,
    position: { x: n.position?.x ?? 0, y: n.position?.y ?? 0 },
    selected: !!n.selected,
    parentId: n.parentId ?? undefined,
    data: n.data ? fromProtoNodeData(n.data) : ({} as DynamicNodeData),
    style:
      n.width && n.height ? { width: n.width, height: n.height } : undefined,
  } as AppNode;

  if (!isStandardSpecial && node.type === "dynamic") {
    node.data.typeId = rawType;
  }

  if (n.width && n.height) {
    node.measured = { width: n.width, height: n.height };
  }

  return node;
}

export function fromProtoNodeData(
  data: flowcraft_proto.v1.INodeData,
): DynamicNodeData {
  return {
    label: data.label ?? "",
    modes: (data.availableModes ?? []) as unknown as RenderMode[],
    activeMode: (data.activeMode ??
      RenderModeVal.MODE_WIDGETS) as unknown as RenderMode,
    media: data.media ? fromProtoMedia(data.media) : undefined,
    widgets: (data.widgets ?? []).map(fromProtoWidget),
    inputPorts: data.inputPorts ?? [],
    outputPorts: data.outputPorts ?? [],
    // Handlers will be attached during hydration
    onChange: () => {
      // Intentionally empty default implementation
    },
  };
}

export function fromProtoMedia(
  media: flowcraft_proto.v1.IMediaContent,
): MediaDef {
  return {
    type: media.type as unknown as MediaType,
    url: media.url ?? undefined,
    content: media.content ?? undefined,
    galleryUrls: media.galleryUrls ?? [],
  };
}

export function fromProtoWidget(widget: flowcraft_proto.v1.IWidget): WidgetDef {
  let value: unknown = undefined;
  try {
    value = widget.valueJson ? JSON.parse(widget.valueJson) : undefined;
  } catch (e) {
    console.error("Failed to parse widget valueJson", e);
  }

  return {
    id: widget.id ?? "",
    type: widget.type as unknown as WidgetType,
    label: widget.label ?? "",
    value: value,
    options: (widget.options ?? []).map((o) => ({
      label: o.label ?? "",
      value: o.value ?? "",
    })),
    config: {
      min: widget.config?.min ?? undefined,
      max: widget.config?.max ?? undefined,
      step: widget.config?.step ?? undefined,
      dynamicOptions: !!widget.config?.dynamicOptions,
    },
    inputPortId: widget.inputPortId ?? undefined,
  };
}
