import React from "react";
import { NodeContextMenu } from "./NodeContextMenu";
import { EdgeContextMenu } from "./EdgeContextMenu";
import { GalleryItemContextMenu } from "./GalleryItemContextMenu";
import { PaneContextMenu } from "./PaneContextMenu";
import { AppNodeType } from "../../types";

interface Props {
  contextMenu: any;
  nodes: any[];
  edges: any[];
  templates: any[];
  availableActions: any[];
  onClose: () => void;
  onDeleteNode: (id: string) => void;
  onDeleteEdge: (id: string) => void;
  onOpenEditor: (id: string) => void;
  onCopy: () => void;
  onDuplicate: () => void;
  onGroup: () => void;
  onAutoLayout: () => void;
  onPaste: () => void;
  onAddNode: (t: any) => void;
  onExecuteAction: (a: any) => void;
}

export const ContextMenuOverlay: React.FC<Props> = (props) => {
  if (!props.contextMenu) return null;
  const { contextMenu, nodes, edges } = props;

  return (
    <>
      {(contextMenu.nodeId || nodes.some((n) => n.selected)) && (
        <NodeContextMenu
          x={contextMenu.x}
          y={contextMenu.y}
          nodeId={contextMenu.nodeId || ""}
          onClose={props.onClose}
          onDelete={() => {
            if (contextMenu.nodeId) {
              const node = nodes.find((n) => n.id === contextMenu.nodeId);
              if (node?.selected) {
                nodes
                  .filter((n) => n.selected)
                  .forEach((n) => {
                    props.onDeleteNode(n.id);
                  });
                edges
                  .filter((e) => e.selected)
                  .forEach((e) => {
                    props.onDeleteEdge(e.id);
                  });
              } else {
                props.onDeleteNode(contextMenu.nodeId);
              }
            } else {
              nodes
                .filter((n) => n.selected)
                .forEach((n) => {
                  props.onDeleteNode(n.id);
                });
            }
            props.onClose();
          }}
          onFocus={() => {}}
          onOpenEditor={() => {
            if (contextMenu.nodeId) props.onOpenEditor(contextMenu.nodeId);
            props.onClose();
          }}
          onCopy={props.onCopy}
          onDuplicate={props.onDuplicate}
          onGroupSelected={
            nodes.some((n) => n.selected) ? props.onGroup : undefined
          }
          onLayoutGroup={
            nodes.find((n) => n.id === contextMenu.nodeId)?.type ===
            AppNodeType.GROUP
              ? props.onAutoLayout
              : undefined
          }
          dynamicActions={props.availableActions.map((a) => ({
            id: a.id,
            name: a.label,
            path: a.path,
            onClick: () => {
              props.onExecuteAction(a);
            },
          }))}
        />
      )}
      {contextMenu.edgeId && !nodes.some((n) => n.selected) && (
        <EdgeContextMenu
          x={contextMenu.x}
          y={contextMenu.y}
          edgeId={contextMenu.edgeId}
          onClose={props.onClose}
          onDelete={() => {
            const edge = edges.find((e) => e.id === contextMenu.edgeId);
            if (edge?.selected) {
              nodes
                .filter((n) => n.selected)
                .forEach((n) => {
                  props.onDeleteNode(n.id);
                });
              edges
                .filter((e) => e.selected)
                .forEach((e) => {
                  props.onDeleteEdge(e.id);
                });
            } else {
              props.onDeleteEdge(contextMenu.edgeId);
            }
            props.onClose();
          }}
        />
      )}
      {contextMenu.galleryItemUrl && (
        <GalleryItemContextMenu
          x={contextMenu.x}
          y={contextMenu.y}
          url={contextMenu.galleryItemUrl}
          onClose={props.onClose}
          onExtract={(url) => {
            console.log("Extract", url);
            props.onClose();
          }}
        />
      )}
      {!contextMenu.nodeId &&
        !contextMenu.edgeId &&
        !contextMenu.galleryItemUrl &&
        !nodes.some((n) => n.selected) && (
          <PaneContextMenu
            x={contextMenu.x}
            y={contextMenu.y}
            templates={props.templates}
            onAddNode={props.onAddNode}
            onAutoLayout={props.onAutoLayout}
            onClose={props.onClose}
            onPaste={props.onPaste}
            onCopy={nodes.some((n) => n.selected) ? props.onCopy : undefined}
            onDuplicate={
              nodes.some((n) => n.selected) ? props.onDuplicate : undefined
            }
            onGroupSelected={
              nodes.some((n) => n.selected) ? props.onGroup : undefined
            }
            onDeleteSelected={
              nodes.some((n) => n.selected) || edges.some((e) => e.selected)
                ? () => {
                    nodes
                      .filter((n) => n.selected)
                      .forEach((n) => {
                        props.onDeleteNode(n.id);
                      });
                    edges
                      .filter((e) => e.selected)
                      .forEach((e) => {
                        props.onDeleteEdge(e.id);
                      });
                    props.onClose();
                  }
                : undefined
            }
          />
        )}
    </>
  );
};
