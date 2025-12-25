import React, { useState, useEffect } from "react";
import {
  withNodeHandlers,
  type NodeRendererProps,
} from "./hocs/withNodeHandlers";
import {
  type DynamicNodeType,
  type WidgetDef,
  type DynamicNodeData,
} from "../types";
import { WidgetWrapper } from "./widgets/WidgetWrapper";
import { TextField } from "./widgets/TextField";
import { SelectField } from "./widgets/SelectField";
import { CheckboxField } from "./widgets/CheckboxField";
import { SliderField } from "./widgets/SliderField";
import { ImageRenderer } from "./media/ImageRenderer";
import { VideoRenderer } from "./media/VideoRenderer";
import { AudioRenderer } from "./media/AudioRenderer";
import { MarkdownRenderer } from "./media/MarkdownRenderer";
import { GalleryWrapper } from "./media/GalleryWrapper";
import { useFlowStore } from "../store/flowStore";
import { useMockSocket } from "../hooks/useMockSocket";
import { useTaskStore } from "../store/taskStore";
import { v4 as uuidv4 } from "uuid";
import type { AppNode } from "../types";
import { WidgetType, MediaType, PortStyle } from "../types";
import { useStore } from "@xyflow/react";
import { PortHandle } from "./base/PortHandle";
import { flowcraft } from "../generated/flowcraft";

const PortLabelRow: React.FC<{
  nodeId: string;
  inputPort?: flowcraft.v1.IPort;
  outputPort?: flowcraft.v1.IPort;
}> = ({ nodeId, inputPort, outputPort }) => {
  const edges = useStore((s) => s.edges);
  const isInputConnected = inputPort
    ? edges.some((e) => e.target === nodeId && e.targetHandle === inputPort.id)
    : false;
  const isOutputConnected = outputPort
    ? edges.some((e) => e.source === nodeId && e.sourceHandle === outputPort.id)
    : false;

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        height: "24px",
        position: "relative",
        width: "100%",
        boxSizing: "border-box",
        justifyContent: "space-between",
        padding: "0 12px",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "8px",
          flex: 1,
          minWidth: 0,
          position: "relative",
          height: "100%",
        }}
      >
        {inputPort && (
          <>
            <PortHandle
              nodeId={nodeId}
              portId={inputPort.id!}
              type="target"
              style={(inputPort.style as PortStyle) || undefined}
              mainType={inputPort.type?.mainType || undefined}
              itemType={inputPort.type?.itemType || undefined}
              isGeneric={!!inputPort.type?.isGeneric}
              color={inputPort.color || undefined}
              description={inputPort.description || undefined}
              sideOffset={12}
            />
            <div
              style={{
                fontSize: "11px",
                fontWeight: 600,
                color: "var(--text-color)",
                whiteSpace: "nowrap",
                overflow: "hidden",
                textOverflow: "ellipsis",
                opacity: isInputConnected ? 1 : 0.6,
              }}
            >
              {inputPort.label}
            </div>
          </>
        )}
      </div>

      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "8px",
          flex: 1,
          minWidth: 0,
          justifyContent: "flex-end",
          textAlign: "right",
          position: "relative",
          height: "100%",
        }}
      >
        {outputPort && (
          <>
            <div
              style={{
                fontSize: "11px",
                fontWeight: 600,
                color: "var(--text-color)",
                whiteSpace: "nowrap",
                overflow: "hidden",
                textOverflow: "ellipsis",
                opacity: isOutputConnected ? 1 : 0.6,
              }}
            >
              {outputPort.label}
            </div>
            <PortHandle
              nodeId={nodeId}
              portId={outputPort.id!}
              type="source"
              style={(outputPort.style as PortStyle) || undefined}
              mainType={outputPort.type?.mainType || undefined}
              itemType={outputPort.type?.itemType || undefined}
              isGeneric={!!outputPort.type?.isGeneric}
              color={outputPort.color || undefined}
              description={outputPort.description || undefined}
              sideOffset={12}
            />
          </>
        )}
      </div>
    </div>
  );
};

const WidgetRenderer: React.FC<{
  nodeId: string;
  widget: WidgetDef;
  onValueChange: (val: unknown) => void;
  onClick: () => void;
}> = ({ nodeId, widget, onValueChange, onClick }) => {
  const {
    sendWidgetUpdate,
    fetchWidgetOptions,
    streamAction,
    executeTask,
    cancelTask,
  } = useMockSocket({ disablePolling: true });
  const { addNode } = useFlowStore.getState();

  const handleValueChange = (val: unknown) => {
    onValueChange(val);
    sendWidgetUpdate(nodeId, widget.id, val);
  };

  let component;
  switch (widget.type) {
    case WidgetType.WIDGET_TEXT:
      component = (
        <TextField
          value={widget.value as string}
          onChange={handleValueChange}
          label={widget.label}
        />
      );
      break;
    case WidgetType.WIDGET_SELECT:
      component = (
        <SelectField
          value={widget.value}
          onChange={handleValueChange}
          label={widget.label}
          options={widget.options || []}
          onFetchOptions={() => fetchWidgetOptions(nodeId, widget.id)}
        />
      );
      break;
    case WidgetType.WIDGET_CHECKBOX:
      component = (
        <CheckboxField
          value={!!widget.value}
          onChange={handleValueChange}
          label={widget.label}
        />
      );
      break;
    case WidgetType.WIDGET_SLIDER:
      component = (
        <SliderField
          value={widget.value as number}
          onChange={handleValueChange}
          label={widget.label}
          min={widget.config?.min as number}
          max={widget.config?.max as number}
        />
      );
      break;
    case WidgetType.WIDGET_BUTTON:
      component = (
        <button
          className="nodrag"
          onClick={() => {
            onClick();
            const val = typeof widget.value === "string" ? widget.value : "";
            if (val.startsWith("stream-to:")) {
              const targetWidgetId = val.split(":")[1];
              let currentBuffer = "";
              streamAction(nodeId, widget.id, (chunk) => {
                currentBuffer += chunk;
                const store = useFlowStore.getState();
                const node = store.nodes.find((n) => n.id === nodeId);
                if (node && node.type === "dynamic" && node.data.widgets) {
                  const updatedWidgets = (node.data.widgets as WidgetDef[]).map(
                    (w) =>
                      w.id === targetWidgetId
                        ? { ...w, value: currentBuffer }
                        : w,
                  );
                  store.updateNodeData(nodeId, { widgets: updatedWidgets });
                }
              });
            }
            if (val.startsWith("task:")) {
              const taskType = val.split(":")[1];
              const taskId = uuidv4();
              const parentNode = useFlowStore
                .getState()
                .nodes.find((n) => n.id === nodeId);
              const position = parentNode
                ? { x: parentNode.position.x + 300, y: parentNode.position.y }
                : { x: 0, y: 0 };
              const placeholderNode: AppNode = {
                id: `task-${taskId}`,
                type: "processing",
                position,
                data: {
                  label: `Running ${taskType}...`,
                  taskId,
                  onCancel: (tid: string) => cancelTask(tid),
                },
              } as AppNode;
              addNode(placeholderNode);
              useTaskStore.getState().registerTask(taskId);
              executeTask(taskId, taskType, { sourceNodeId: nodeId });
            }
          }}
          style={{ width: "100%", padding: "4px" }}
        >
          {widget.label}
        </button>
      );
      break;
    default:
      return null;
  }
  return <div>{component}</div>;
};

const RenderMedia: React.FC<
  NodeRendererProps<DynamicNodeType> & {
    onOverflowChange?: (o: "visible" | "hidden") => void;
  }
> = (props) => {
  const { id, data, onOverflowChange } = props;
  const dispatchNodeEvent = useFlowStore((state) => state.dispatchNodeEvent);
  if (!data.media) return null;
  const layoutProps = props as unknown as {
    width?: number;
    measured?: { width: number; height: number };
    style?: React.CSSProperties;
  };
  const nodeWidth =
    layoutProps.width ??
    layoutProps.measured?.width ??
    (layoutProps.style?.width as number) ??
    240;

  /* eslint-disable @typescript-eslint/no-explicit-any */
  const nodeHeight =
    (layoutProps as any).height ??
    (layoutProps as any).measured?.height ??
    (layoutProps as any).style?.height ??
    180;
  /* eslint-enable @typescript-eslint/no-explicit-any */

  const handleOpenPreview = (index: number = 0) => {
    dispatchNodeEvent("open-preview", { nodeId: id, index });
  };

  const renderContent = (url: string, type: MediaType, index: number = 0) => {
    switch (type) {
      case MediaType.MEDIA_IMAGE:
        return (
          <div
            onDoubleClick={() => handleOpenPreview(index)}
            style={{ width: "100%", height: "100%" }}
          >
            <ImageRenderer url={url} />
          </div>
        );
      case MediaType.MEDIA_VIDEO:
        return (
          <div
            onDoubleClick={() => handleOpenPreview(index)}
            style={{ width: "100%", height: "100%" }}
          >
            <VideoRenderer url={url} autoPlay />
          </div>
        );
      case MediaType.MEDIA_AUDIO:
        return (
          <div
            onDoubleClick={() => handleOpenPreview(index)}
            style={{ width: "100%", height: "100%" }}
          >
            <AudioRenderer url={url} />
          </div>
        );
      default:
        return (
          <div style={{ padding: "20px", textAlign: "center" }}>
            Unsupported media: {url}
          </div>
        );
    }
  };

  if (data.media.type === MediaType.MEDIA_MARKDOWN) {
    return (
      <MarkdownRenderer
        content={data.media.content || ""}
        onEdit={(newContent) => {
          data.onChange(id, { media: { ...data.media!, content: newContent } });
        }}
      />
    );
  }

  const gallery = data.media.galleryUrls || [];
  return (
    <GalleryWrapper
      id={id}
      nodeWidth={nodeWidth}
      nodeHeight={nodeHeight}
      mainContent={renderContent(data.media.url || "", data.media.type, 0)}
      gallery={gallery}
      mediaType={data.media.type}
      renderItem={(url) =>
        renderContent(url, data.media!.type, gallery.indexOf(url) + 1)
      }
      onGalleryItemContext={(nodeId, url, mediaType, x, y) => {
        data.onGalleryItemContext?.(nodeId, url, mediaType, x, y);
      }}
      onExpand={(expanded) =>
        onOverflowChange?.(expanded ? "visible" : "hidden")
      }
    />
  );
};

const RenderWidgets: React.FC<
  NodeRendererProps<DynamicNodeType> & { onToggleMode: () => void }
> = (props) => {
  const { id, data, onToggleMode, selected } = props;
  const isSwitchable = data.modes.length > 1;
  const { sendNodeUpdate } = useMockSocket({ disablePolling: true });

  const [isEditing, setIsEditing] = useState(false);
  const [localLabel, setLocalLabel] = useState(data.label);

  useEffect(() => {
    if (!isEditing) {
      const t = setTimeout(() => setLocalLabel(data.label), 0);
      return () => clearTimeout(t);
    }
  }, [data.label, isEditing]);

  useEffect(() => {
    if (!selected) {
      const t = setTimeout(() => setIsEditing(false), 0);
      return () => clearTimeout(t);
    }
  }, [selected]);

  const handleExitEdit = () => {
    setIsEditing(false);
    sendNodeUpdate(id, { label: localLabel });
  };

  const inputs = (data.inputPorts as flowcraft.v1.IPort[]) || [];
  const outputs = (data.outputPorts as flowcraft.v1.IPort[]) || [];
  const rowCount = Math.max(inputs.length, outputs.length);
  const portRows = [];
  for (let i = 0; i < rowCount; i++) {
    portRows.push({ input: inputs[i], output: outputs[i] });
  }

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        minWidth: "150px",
        flex: 1,
      }}
    >
      {/* 1. LABEL AREA */}
      <div
        onClick={(e) => {
          if (selected) {
            e.stopPropagation();
            setIsEditing(true);
          }
        }}
        onContextMenu={(e) => {
          if (isEditing) e.stopPropagation();
        }}
        onMouseDown={(e) => {
          if (isEditing) e.stopPropagation();
        }}
        style={{
          marginBottom: "8px",
          borderBottom: "1px solid var(--node-border)",
          padding: "10px 12px",
          display: "flex",
          alignItems: "center",
          minHeight: "38px",
          boxSizing: "border-box",
        }}
      >
        {isEditing ? (
          <input
            className="nodrag nopan"
            autoFocus
            style={{
              background: "var(--input-bg)",
              border: "none",
              color: "var(--text-color)",
              fontSize: "13px",
              fontWeight: "bold",
              width: "100%",
              outline: "none",
              padding: "2px 4px",
              borderRadius: "2px",
            }}
            value={localLabel}
            onChange={(e) => {
              setLocalLabel(e.target.value);
              data.onChange(id, { label: e.target.value });
            }}
            onBlur={handleExitEdit}
            onKeyDown={(e) => {
              e.stopPropagation();
              if (e.key === "Enter") handleExitEdit();
            }}
          />
        ) : (
          <div
            style={{
              fontSize: "13px",
              fontWeight: "bold",
              color: "var(--text-color)",
              userSelect: "text",
              width: "100%",
            }}
          >
            {data.label}
          </div>
        )}
      </div>

      <div style={{ display: "flex", flexDirection: "column" }}>
        {portRows.map((row, idx) => (
          <PortLabelRow
            key={idx}
            nodeId={id}
            inputPort={row.input}
            outputPort={row.output}
          />
        ))}
      </div>

      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: "8px",
          padding: "8px 12px",
          position: "relative",
        }}
      >
        {(data.widgets as WidgetDef[] | undefined)?.map((w) => (
          <div key={w.id} style={{ position: "relative", width: "100%" }}>
            {w.inputPortId && (
              <PortHandle
                nodeId={id}
                portId={w.inputPortId}
                type="target"
                sideOffset={12}
                style={PortStyle.PORT_STYLE_CIRCLE}
                color="var(--primary-color)"
              />
            )}
            <WidgetWrapper
              isSwitchable={isSwitchable}
              onToggleMode={onToggleMode}
              inputPortId={w.inputPortId}
              nodeId={id}
            >
              <WidgetRenderer
                nodeId={id}
                widget={w}
                onValueChange={(val) => {
                  const updatedWidgets = (data.widgets as WidgetDef[]).map(
                    (item) =>
                      item.id === w.id ? { ...item, value: val } : item,
                  );
                  data.onChange(id, { widgets: updatedWidgets });
                }}
                onClick={() => data.onWidgetClick?.(id, w.id)}
              />
            </WidgetWrapper>
          </div>
        ))}
      </div>
    </div>
  );
};

export const DynamicNode = withNodeHandlers<DynamicNodeData, DynamicNodeType>(
  RenderMedia,
  RenderWidgets,
);
