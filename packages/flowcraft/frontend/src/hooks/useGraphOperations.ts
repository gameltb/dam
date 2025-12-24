import { useState, useCallback } from "react";
import { v4 as uuidv4 } from "uuid";
import toast from "react-hot-toast";
import type { AppNode, NodeData } from "../types";
import { useFlowStore } from "../store/flowStore";
import dagre from "dagre";
import { type Edge } from "@xyflow/react";

interface UseGraphOperationsProps {
  clientVersion: number;
  sendJsonMessage: (message: any) => void; // eslint-disable-line @typescript-eslint/no-explicit-any
}

export const useGraphOperations = ({
  clientVersion,
  sendJsonMessage,
}: UseGraphOperationsProps) => {
  const {
    nodes,
    edges,
    setNodes,
    setEdges,
    addNode: addNodeToStore,
  } = useFlowStore();

  const [isFocusView, setFocusView] = useState(false);
  const [originalNodes, setOriginalNodes] = useState<AppNode[] | null>(null);

  const incrementalLayout = useCallback(
    (currentNodes: AppNode[], currentEdges: Edge[], newNodesIds: string[]) => {
      const updatedNodes = [...currentNodes];

      newNodesIds.forEach((newNodeId) => {
        const nodeIdx = updatedNodes.findIndex((n) => n.id === newNodeId);
        if (nodeIdx === -1) return;

        const newNode = updatedNodes[nodeIdx];

        // Get dimensions
        const nodeWidth =
          newNode.measured?.width || (newNode.style?.width as number) || 250;
        const nodeHeight =
          newNode.measured?.height || (newNode.style?.height as number) || 150;
        const PADDING_X = 100;
        const PADDING_Y = 50;

        // Find incoming edges to this new node
        const incomingEdges = currentEdges.filter(
          (e) => e.target === newNodeId,
        );
        const sourceNodes = incomingEdges
          .map((e) => updatedNodes.find((n) => n.id === e.source))
          .filter(Boolean) as AppNode[];

        let targetX = newNode.position.x;
        let targetY = newNode.position.y;

        if (sourceNodes.length > 0) {
          // Position to the right of the sources
          const maxX = Math.max(
            ...sourceNodes.map((s) => {
              const sw = s.measured?.width || (s.style?.width as number) || 200;
              return s.position.x + sw;
            }),
          );
          const avgY =
            sourceNodes.reduce((sum, s) => sum + s.position.y, 0) /
            sourceNodes.length;

          targetX = maxX + PADDING_X;
          targetY = avgY;
        } else {
          targetX = 100;
          targetY = 100;
        }

        const initialTargetY = targetY;
        let foundSpot = false;

        // Search offsets: 0, +1, -1, +2, -2, +3, -3
        const searchOffsets = [0, 1, -1, 2, -2, 3, -3];

        for (const offsetMultiplier of searchOffsets) {
          const candidateY =
            initialTargetY + offsetMultiplier * (nodeHeight + PADDING_Y);

          const collision = updatedNodes.some((otherNode) => {
            if (otherNode.id === newNodeId) return false;
            const ow =
              otherNode.measured?.width ||
              (otherNode.style?.width as number) ||
              200;
            const oh =
              otherNode.measured?.height ||
              (otherNode.style?.height as number) ||
              100;

            return (
              targetX < otherNode.position.x + ow &&
              targetX + nodeWidth > otherNode.position.x &&
              candidateY < otherNode.position.y + oh &&
              candidateY + nodeHeight > otherNode.position.y
            );
          });

          if (!collision) {
            targetY = candidateY;
            foundSpot = true;
            break;
          }
        }

        if (!foundSpot) {
          targetY = initialTargetY;
        }

        updatedNodes[nodeIdx] = {
          ...newNode,
          position: { x: targetX, y: targetY },
        };
      });

      setNodes(updatedNodes);
      return updatedNodes;
    },
    [setNodes],
  );

  const groupSelectedNodes = useCallback(() => {
    const selectedNodes = nodes.filter((n) => n.selected);
    if (selectedNodes.length < 2) {
      toast.error("Please select at least 2 nodes to group.");
      return;
    }

    const minX = Math.min(...selectedNodes.map((n) => n.position.x));
    const minY = Math.min(...selectedNodes.map((n) => n.position.y));
    const maxX = Math.max(
      ...selectedNodes.map((n) => n.position.x + (n.measured?.width || 200)),
    );
    const maxY = Math.max(
      ...selectedNodes.map((n) => n.position.y + (n.measured?.height || 100)),
    );

    const padding = 40;
    const groupId = uuidv4();

    const groupNode: AppNode = {
      id: groupId,
      type: "groupNode",
      position: { x: minX - padding, y: minY - padding },
      style: {
        width: maxX - minX + padding * 2,
        height: maxY - minY + padding * 2,
      },
      data: { label: "New Group" },
    } as AppNode;

    const updatedNodes = nodes.map((node) => {
      if (node.selected) {
        return {
          ...node,
          parentId: groupId,
          extent: "parent" as const,
          position: {
            x: node.position.x - (minX - padding),
            y: node.position.y - (minY - padding),
          },
          selected: false,
        };
      }
      return node;
    });

    setNodes([groupNode, ...updatedNodes]);
  }, [nodes, setNodes]);

  const layoutGroup = useCallback(
    (groupId: string) => {
      const childNodes = nodes.filter((n) => n.parentId === groupId);
      if (childNodes.length === 0) return;

      const dagreGraph = new dagre.graphlib.Graph();
      dagreGraph.setDefaultEdgeLabel(() => ({}));
      dagreGraph.setGraph({ rankdir: "LR", nodesep: 50, ranksep: 70 });

      childNodes.forEach((node) => {
        dagreGraph.setNode(node.id, {
          width: node.measured?.width || 200,
          height: node.measured?.height || 100,
        });
      });

      // Only consider edges between children of this group
      const childIds = new Set(childNodes.map((n) => n.id));
      const groupEdges = edges.filter(
        (e) => childIds.has(e.source) && childIds.has(e.target),
      );

      groupEdges.forEach((edge) => {
        dagreGraph.setEdge(edge.source, edge.target);
      });

      dagre.layout(dagreGraph);

      const padding = 40;
      const layoutedNodes = nodes.map((node) => {
        if (node.parentId === groupId) {
          const nodeWithPosition = dagreGraph.node(node.id);
          return {
            ...node,
            position: {
              x:
                nodeWithPosition.x -
                (node.measured?.width || 200) / 2 +
                padding,
              y:
                nodeWithPosition.y -
                (node.measured?.height || 100) / 2 +
                padding,
            },
          };
        }
        return node;
      });

      // Adjust group size
      const groupMaxX = Math.max(
        ...layoutedNodes
          .filter((n) => n.parentId === groupId)
          .map((n) => n.position.x + (n.measured?.width || 200)),
      );
      const groupMaxY = Math.max(
        ...layoutedNodes
          .filter((n) => n.parentId === groupId)
          .map((n) => n.position.y + (n.measured?.height || 100)),
      );

      const finalNodes = layoutedNodes.map((node) => {
        if (node.id === groupId) {
          return {
            ...node,
            style: {
              ...node.style,
              width: groupMaxX + padding,
              height: groupMaxY + padding,
            },
          };
        }
        return node;
      });

      setNodes(finalNodes);
    },
    [nodes, edges, setNodes],
  );

  const autoLayout = useCallback(
    (currentNodes: AppNode[] = nodes, currentEdges: Edge[] = edges) => {
      const dagreGraph = new dagre.graphlib.Graph();

      dagreGraph.setDefaultEdgeLabel(() => ({}));

      // Set graph direction to Left-to-Right

      dagreGraph.setGraph({ rankdir: "LR", nodesep: 70, ranksep: 100 });

      // Filter: Only include top-level nodes and group nodes themselves

      const topLevelNodes = currentNodes.filter((node) => !node.parentId);

      topLevelNodes.forEach((node) => {
        const nw = node.measured?.width || (node.style?.width as number) || 200;

        const nh =
          node.measured?.height || (node.style?.height as number) || 100;

        dagreGraph.setNode(node.id, { width: nw, height: nh });
      });

      // Filter edges: Only include edges where both source and target are in our top-level set

      const topLevelNodeIds = new Set(topLevelNodes.map((n) => n.id));

      currentEdges.forEach((edge) => {
        if (
          topLevelNodeIds.has(edge.source) &&
          topLevelNodeIds.has(edge.target)
        ) {
          dagreGraph.setEdge(edge.source, edge.target);
        }
      });

      dagre.layout(dagreGraph);

      const layoutedNodes = currentNodes.map((node) => {
        // If it's a child node, we don't change its internal relative position

        if (node.parentId) return node;

        const nodeWithPosition = dagreGraph.node(node.id);

        const nw = node.measured?.width || (node.style?.width as number) || 200;

        const nh =
          node.measured?.height || (node.style?.height as number) || 100;

        return {
          ...node,

          position: {
            x: nodeWithPosition.x - nw / 2, // Accurate center alignment

            y: nodeWithPosition.y - nh / 2,
          },
        };
      });

      setNodes(layoutedNodes);

      return layoutedNodes;
    },

    [nodes, edges, setNodes],
  );

  const addNode = useCallback(
    (
      type: string,
      data: NodeData,
      position: { x: number; y: number },
    ): AppNode => {
      const newNode: AppNode = {
        id: uuidv4(),
        type,
        position,
        data,
      } as AppNode;
      addNodeToStore(newNode);
      return newNode;
    },
    [addNodeToStore],
  );

  const deleteEdge = useCallback(
    (edgeId: string) => {
      setEdges(edges.filter((e) => e.id !== edgeId));
    },
    [edges, setEdges],
  );

  const deleteNode = useCallback(
    (nodeId: string) => {
      setNodes(nodes.filter((n) => n.id !== nodeId));
      setEdges(edges.filter((e) => e.source !== nodeId && e.target !== nodeId));
    },
    [nodes, edges, setNodes, setEdges],
  );

  const focusNode = useCallback(
    (nodeId: string) => {
      setOriginalNodes(nodes);
      const focusedNode = nodes.find((n) => n.id === nodeId);
      if (!focusedNode) return;

      const connectedEdges = edges.filter(
        (e) => e.source === focusedNode.id || e.target === focusedNode.id,
      );
      const neighborIds = new Set(
        connectedEdges.flatMap((e) => [e.source, e.target]),
      );
      const focusNodes = nodes.filter((n) => neighborIds.has(n.id));

      const center = { x: 250, y: 250 };
      const radius = 200;
      const arrangedNodes = focusNodes.map((node, i) => {
        if (node.id === focusedNode.id) {
          return { ...node, position: center };
        }
        const angle = (i / (focusNodes.length - 1)) * 2 * Math.PI;
        return {
          ...node,
          position: {
            x: center.x + radius * Math.cos(angle),
            y: center.y + radius * Math.sin(angle),
          },
        };
      });

      setNodes(arrangedNodes);
      setFocusView(true);
    },
    [nodes, edges, setNodes],
  );

  const exitFocusView = useCallback(() => {
    if (originalNodes) {
      const focusedNodeIds = new Set(nodes.map((n) => n.id));
      const updatedOriginalNodes = originalNodes.map((originalNode) => {
        if (focusedNodeIds.has(originalNode.id)) {
          const focusedNode = nodes.find((n) => n.id === originalNode.id);
          return focusedNode || originalNode;
        }
        return originalNode;
      });
      setNodes(updatedOriginalNodes);
      sendJsonMessage({
        type: "sync_graph",
        payload: {
          version: clientVersion,
          graph: { nodes: updatedOriginalNodes, edges },
        },
      });
    }
    setFocusView(false);
    setOriginalNodes(null);
  }, [nodes, edges, originalNodes, clientVersion, sendJsonMessage, setNodes]);

  return {
    addNode,
    deleteNode,
    deleteEdge,
    focusNode,
    exitFocusView,
    autoLayout,
    incrementalLayout,
    groupSelectedNodes,
    layoutGroup,
    isFocusView,
  };
};
