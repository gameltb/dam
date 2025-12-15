import { useCallback, useEffect, useMemo, useState, memo } from "react";
import { useTheme } from "./ThemeContext";
import {
  ReactFlow,
  MiniMap,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  addEdge,
  type OnConnect,
  type Node,
  type Edge,
  BackgroundVariant,
  type NodeTypes,
} from "@xyflow/react";
import useWebSocket from "react-use-websocket";
import { v4 as uuidv4 } from "uuid";
import "@xyflow/react/dist/style.css";
import { TextNode, type TextNodeType } from "./components/TextNode";
import { ImageNode, type ImageNodeType } from "./components/ImageNode";
import { ContextMenu } from "./components/ContextMenu";
import { EntityNode, type EntityNodeType } from "./components/EntityNode";
import {
  ComponentNode,
  type ComponentNodeType,
} from "./components/ComponentNode";
import { StatusPanel } from "./components/StatusPanel";
import { EditUrlModal } from "./components/EditUrlModal";

type NodeData =
  | TextNodeType["data"]
  | ImageNodeType["data"]
  | EntityNodeType["data"]
  | ComponentNodeType["data"];

type AppNode =
  | TextNodeType
  | ImageNodeType
  | EntityNodeType
  | ComponentNodeType;

function App() {
  const [nodes, setNodes, onNodesChange] = useNodesState<AppNode>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [contextMenu, setContextMenu] = useState<{
    x: number;
    y: number;
    nodeId?: string;
  } | null>(null);
  const [isFocusView, setFocusView] = useState(false);
  const [originalNodes, setOriginalNodes] = useState<AppNode[] | null>(null);
  const { theme, toggleTheme } = useTheme();
  const [wsUrl, setWsUrl] = useState("ws://127.0.0.1:8000/ws");
  const [isModalOpen, setIsModalOpen] = useState(false);

  const { sendJsonMessage, lastJsonMessage, readyState } = useWebSocket(wsUrl, {
    share: true,
  });

  const connectionStatus = {
    [0]: 'Connecting',
    [1]: 'Connected',
    [2]: 'Closing',
    [3]: 'Disconnected',
  }[readyState];

  const handleNodeDataChange = useCallback(
    (nodeId: string, data: Partial<NodeData>) => {
      setNodes((prevNodes) =>
        prevNodes.map((n) =>
          n.id === nodeId
            ? ({ ...n, data: { ...n.data, ...data } } as AppNode)
            : n,
        ),
      );
    },
    [setNodes],
  );

  const memoizedNodeTypes: NodeTypes = useMemo(
    () => ({
      text: (props) => (
        <TextNode
          {...props}
          data={{ ...props.data, onChange: handleNodeDataChange }}
        />
      ),
      image: (props) => (
        <ImageNode
          {...props}
          data={{ ...props.data, onChange: handleNodeDataChange }}
        />
      ),
      entity: (props) => (
        <EntityNode
          {...props}
          data={{ ...props.data, onChange: handleNodeDataChange }}
        />
      ),
      component: (props) => (
        <ComponentNode
          {...props}
          data={{ ...props.data, onChange: handleNodeDataChange }}
        />
      ),
    }),
    [handleNodeDataChange],
  );

  useEffect(() => {
    if (lastJsonMessage) {
      const graph = lastJsonMessage as {
        nodes: AppNode[];
        edges: Edge[];
      };
      setNodes(graph.nodes);
      setEdges(graph.edges);
    }
  }, [lastJsonMessage, setNodes, setEdges]);

  const onConnect: OnConnect = useCallback(
    (params) => {
      const newEdge = {
        ...params,
        id: `e${params.source}-${params.target}`,
      } as Edge;
      setEdges((prevEdges) => addEdge(newEdge, prevEdges));
    },
    [setEdges],
  );

  const addNode = (
    type: "text" | "image" | "entity" | "component",
    data: NodeData,
    position: { x: number; y: number },
  ): AppNode => {
    const newNode = {
      id: uuidv4(),
      type,
      position,
      data,
    };
    setNodes((prevNodes) => [...prevNodes, newNode as AppNode]);
    return newNode as AppNode;
  };

  const onPaneContextMenu = useCallback(
    (event: React.MouseEvent) => {
      event.preventDefault();
      const pane = (event.target as Element).closest('.react-flow__pane');
      if (pane) {
        setContextMenu({
          x: event.clientX,
          y: event.clientY,
        });
      }
    },
    [setContextMenu],
  );

  const onNodeContextMenu = useCallback(
    (event: React.MouseEvent, node: Node) => {
      event.preventDefault();
      setContextMenu({ x: event.clientX, y: event.clientY, nodeId: node.id });
    },
    [setContextMenu],
  );

  const onPaneClick = useCallback(() => setContextMenu(null), [setContextMenu]);

  const handleDelete = () => {
    if (!contextMenu) return;
    setNodes((prevNodes) =>
      prevNodes.filter((n) => n.id !== contextMenu.nodeId),
    );
    setEdges((prevEdges) =>
      prevEdges.filter(
        (e) =>
          e.source !== contextMenu.nodeId && e.target !== contextMenu.nodeId,
      ),
    );
    setContextMenu(null);
  };

  const handleFocus = () => {
    if (!contextMenu) return;
    setOriginalNodes(nodes);
    const focusedNode = nodes.find((n) => n.id === contextMenu.nodeId);
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
    setContextMenu(null);
  };

  const exitFocusView = () => {
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
      sendJsonMessage({ nodes: updatedOriginalNodes, edges });
    }
    setFocusView(false);
    setOriginalNodes(null);
  };

  const onShowDamEntity = async () => {
    const entityId = prompt("Enter DAM Entity ID:");
    if (!entityId) return;

    try {
      const response = await fetch(`/entity/${entityId}`);
      if (!response.ok) {
        throw new Error(`Entity not found: ${response.statusText}`);
      }
      const components = (await response.json()) as Record<string, unknown[]>;

      const entityNode = addNode(
        "entity",
        { entityId, onChange: handleNodeDataChange },
        { x: 250, y: 250 },
      );

      const componentNodes = Object.entries(components).flatMap(
        ([name, comps], i) =>
          comps.map((comp, j) => {
            const yOffset = (Object.keys(components).length / 2 - i) * 150;
            return addNode(
              "component",
              {
                componentName: name,
                json: JSON.stringify(comp, null, 2),
                onChange: handleNodeDataChange,
              },
              { x: 500, y: 250 + yOffset + j * 100 },
            );
          }),
      );

      const newEdges = componentNodes.map((compNode) => ({
        id: uuidv4(),
        source: entityNode.id,
        target: compNode.id,
      }));

      setEdges((prevEdges) => [...prevEdges, ...newEdges]);
    } catch (error) {
      alert(`Error fetching entity: ${error}`);
    }
  };

  useEffect(() => {
    sendJsonMessage({ nodes, edges });
  }, [nodes, edges, sendJsonMessage]);

  return (
    <div style={{ width: "100vw", height: "100vh" }}>
      {isFocusView && (
        <div style={{ position: "absolute", top: 10, left: 10, zIndex: 4 }}>
          <button onClick={exitFocusView}>Back to Global View</button>
        </div>
      )}
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeContextMenu={onNodeContextMenu}
        onPaneContextMenu={onPaneContextMenu}
        onPaneClick={onPaneClick}
        nodeTypes={memoizedNodeTypes}
        fitView
        proOptions={{ hideAttribution: true }}
      >
        <Controls />
        <MiniMap />
        <Background variant={BackgroundVariant.Dots} gap={12} size={1} />
      </ReactFlow>
      {contextMenu && (
        <ContextMenu
          x={contextMenu.x}
          y={contextMenu.y}
          onClose={() => setContextMenu(null)}
          onDelete={contextMenu.nodeId ? handleDelete : undefined}
          onFocus={contextMenu.nodeId ? handleFocus : undefined}
          onShowDamEntity={contextMenu.nodeId ? onShowDamEntity : undefined}
          onToggleTheme={toggleTheme}
          onAddTextNode={() =>
            addNode(
              "text",
              { label: "New Text Node", onChange: handleNodeDataChange },
              { x: contextMenu.x, y: contextMenu.y },
            )
          }
          onAddImageNode={() =>
            addNode(
              "image",
              { url: "", onChange: handleNodeDataChange },
              { x: contextMenu.x, y: contextMenu.y },
            )
          }
          isPaneMenu={!contextMenu.nodeId}
        />
      )}
      <StatusPanel status={connectionStatus} url={wsUrl} onClick={() => setIsModalOpen(true)} />
      {isModalOpen && (
        <EditUrlModal
          currentUrl={wsUrl}
          onClose={() => setIsModalOpen(false)}
          onSave={(newUrl) => setWsUrl(newUrl)}
        />
      )}
    </div>
  );
}

export default memo(App);
