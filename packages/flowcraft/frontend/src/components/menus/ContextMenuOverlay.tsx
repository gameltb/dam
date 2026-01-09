import { type Edge } from "@xyflow/react";
import React from "react";

import { type ActionTemplate } from "@/generated/flowcraft/v1/core/action_pb";
import { type NodeTemplate } from "@/generated/flowcraft/v1/core/node_pb";
import { type AppNode, AppNodeType, type MediaType } from "@/types";

import { EdgeContextMenu } from "./EdgeContextMenu";
import { GalleryItemContextMenu } from "./GalleryItemContextMenu";
import { NodeContextMenu } from "./NodeContextMenu";
import { PaneContextMenu } from "./PaneContextMenu";

interface Props {
  availableActions: ActionTemplate[];
  contextMenu: null | {
    edgeId?: string;
    galleryItemType?: MediaType;
    galleryItemUrl?: string;
    nodeId?: string;
    x: number;
    y: number;
  };
  edges: Edge[];
  nodes: AppNode[];
  onAddNode: (t: NodeTemplate) => void;
  onAutoLayout: () => void;
  onClose: () => void;
  onCopy: () => void;
  onDeleteEdge: (id: string) => void;
  onDeleteNode: (id: string) => void;
  onDuplicate: () => void;
  onExecuteAction: (a: ActionTemplate) => void;
  onGroup: () => void;
  onOpenEditor: (id: string) => void;
  onPaste: () => void;
  templates: NodeTemplate[];
}

export const ContextMenuOverlay: React.FC<Props> = (props) => {
  if (!props.contextMenu) return null;
  const { contextMenu, edges, nodes } = props;

  return (
    <>
      {(contextMenu.nodeId ?? nodes.some((n) => n.selected)) && (
        <NodeContextMenu
          dynamicActions={props.availableActions.map((a) => ({
            id: a.id,
            name: a.label,
            onClick: () => {
              props.onExecuteAction(a);
            },
            path: a.path,
          }))}
          nodeId={contextMenu.nodeId ?? ""}
          onClose={props.onClose}
          onCopy={props.onCopy}
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
          onDuplicate={props.onDuplicate}
          onFocus={() => {
            /* empty */
          }}
          onGroupSelected={
            nodes.some((n) => n.selected) ? props.onGroup : undefined
          }
          onLayoutGroup={
            nodes.find((n) => n.id === contextMenu.nodeId)?.type ===
            AppNodeType.GROUP
              ? props.onAutoLayout
              : undefined
          }
          onOpenEditor={() => {
            if (contextMenu.nodeId) props.onOpenEditor(contextMenu.nodeId);
            props.onClose();
          }}
          x={contextMenu.x}
          y={contextMenu.y}
        />
      )}
      {contextMenu.edgeId && !nodes.some((n) => n.selected) && (
        <EdgeContextMenu
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
            } else if (contextMenu.edgeId) {
              props.onDeleteEdge(contextMenu.edgeId);
            }
            props.onClose();
          }}
          x={contextMenu.x}
          y={contextMenu.y}
        />
      )}
      {contextMenu.galleryItemUrl && (
        <GalleryItemContextMenu
          onClose={props.onClose}
          onExtract={(url) => {
            console.log("Extract", url);
            props.onClose();
          }}
          url={contextMenu.galleryItemUrl}
          x={contextMenu.x}
          y={contextMenu.y}
        />
      )}
      {!contextMenu.nodeId &&
        !contextMenu.edgeId &&
        !contextMenu.galleryItemUrl &&
        !nodes.some((n) => n.selected) && (
          <PaneContextMenu
            onAddNode={props.onAddNode}
            onAutoLayout={props.onAutoLayout}
            onClose={props.onClose}
            onCopy={nodes.some((n) => n.selected) ? props.onCopy : undefined}
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
            onDuplicate={
              nodes.some((n) => n.selected) ? props.onDuplicate : undefined
            }
            onGroupSelected={
              nodes.some((n) => n.selected) ? props.onGroup : undefined
            }
            onPaste={props.onPaste}
            templates={props.templates}
            x={contextMenu.x}
            y={contextMenu.y}
          />
        )}
    </>
  );
};
