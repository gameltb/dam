import React from "react";
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
import { MarkdownRenderer } from "./media/MarkdownRenderer";
import { GalleryWrapper } from "./media/GalleryWrapper";
import { useFlowStore } from "../store/flowStore";

const WidgetRenderer: React.FC<{
  widget: WidgetDef;
  onValueChange: (val: unknown) => void;
  onClick: () => void;
}> = ({ widget, onValueChange, onClick }) => {
  let component;
  switch (widget.type) {
    case "text":
      component = (
        <TextField
          value={widget.value as string}
          onChange={onValueChange}
          label={widget.label}
        />
      );
      break;
    case "select":
      component = (
        <SelectField
          value={widget.value}
          onChange={onValueChange}
          label={widget.label}
          options={widget.options || []}
        />
      );
      break;
    case "checkbox":
      component = (
        <CheckboxField
          value={!!widget.value}
          onChange={onValueChange}
          label={widget.label}
        />
      );
      break;
    case "slider":
      component = (
        <SliderField
          value={widget.value as number}
          onChange={onValueChange}
          label={widget.label}
          min={widget.config?.min as number}
          max={widget.config?.max as number}
        />
      );
      break;
    case "button":
      component = (
        <button onClick={onClick} style={{ width: "100%", padding: "4px" }}>
          {widget.label}
        </button>
      );
      break;
    default:
      return null;
  }

  return <div>{component}</div>;
};

const RenderMedia: React.FC<NodeRendererProps<DynamicNodeType>> = (props) => {
  const { id, data } = props;
  const dispatchNodeEvent = useFlowStore((state) => state.dispatchNodeEvent);

  if (!data.media) return null;

  const layoutProps = props as typeof props & {
    width?: number;
    height?: number;
    measured?: { width: number; height: number };
    style?: React.CSSProperties;
  };

  const nodeWidth =
    layoutProps.width ??
    layoutProps.measured?.width ??
    (layoutProps.style?.width as number) ??
    240;
  const nodeHeight =
    layoutProps.height ??
    layoutProps.measured?.height ??
    (layoutProps.style?.height as number) ??
    180;

  const handleOpenPreview = (index: number = 0) => {
    dispatchNodeEvent("open-preview", { nodeId: id, index });
  };

  const renderContent = (url: string, type: string, index: number = 0) => {
    switch (type) {
      case "image":
        return (
          <div
            onDoubleClick={() => handleOpenPreview(index)}
            style={{ width: "100%", height: "100%" }}
          >
            <ImageRenderer url={url} />
          </div>
        );
      case "video":
        return (
          <div
            onDoubleClick={() => handleOpenPreview(index)}
            style={{ width: "100%", height: "100%" }}
          >
            <VideoRenderer url={url} autoPlay />
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

  if (data.media.type === "markdown") {
    return (
      <MarkdownRenderer
        content={data.media.content || ""}
        onEdit={(newContent) => {
          data.onChange(id, { media: { ...data.media!, content: newContent } });
        }}
      />
    );
  }

  const gallery = data.media.gallery || [];

  return (
    <GalleryWrapper
      id={id}
      nodeWidth={nodeWidth}
      nodeHeight={nodeHeight}
      mainContent={renderContent(data.media.url || "", data.media.type, 0)}
      gallery={gallery}
      renderItem={(url) =>
        renderContent(url, data.media!.type, gallery.indexOf(url) + 1)
      }
      onGalleryItemContext={data.onGalleryItemContext}
    />
  );
};

const RenderWidgets: React.FC<
  NodeRendererProps<DynamicNodeType> & { onToggleMode: () => void }
> = (props) => {
  const { id, data, onToggleMode } = props;
  const isSwitchable = data.modes.length > 1;

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "8px",
        minWidth: "150px",
      }}
    >
      {data.widgets?.map((w) => (
        <WidgetWrapper
          key={w.id}
          isSwitchable={isSwitchable}
          onToggleMode={onToggleMode}
        >
          <WidgetRenderer
            widget={w}
            onValueChange={(val) => {
              const updatedWidgets = data.widgets?.map((item) =>
                item.id === w.id ? { ...item, value: val } : item,
              );
              data.onChange(id, { widgets: updatedWidgets });
            }}
            onClick={() => data.onWidgetClick?.(id, w.id)}
          />
        </WidgetWrapper>
      ))}
    </div>
  );
};

export const DynamicNode = withNodeHandlers<DynamicNodeData, DynamicNodeType>(
  RenderMedia,
  RenderWidgets,
);
