import { useCallback, useEffect, useMemo, useState, memo } from "react";
import { useTheme } from "./hooks/useTheme";
import {
  ReactFlow,
  MiniMap,
  Controls,
  Background,
  type Node,
  BackgroundVariant,
  type NodeTypes,
} from "@xyflow/react";
import { useFlowStore, useTemporalStore } from "./store/flowStore";
import { v4 as uuidv4 } from "uuid";
import "@xyflow/react/dist/style.css";
import { useMockSocket } from "./hooks/useMockSocket";
import { TextNode } from "./components/TextNode";
import { ImageNode } from "./components/ImageNode";
import { ContextMenu } from "./components/ContextMenu";
import { EntityNode } from "./components/EntityNode";
import { ComponentNode } from "./components/ComponentNode";
import { StatusPanel } from "./components/StatusPanel";
import { EditUrlModal } from "./components/EditUrlModal";
import { type NodeData, type AppNode } from "./types";

function App() {
  const {
    nodes,
    edges,
    onNodesChange,
    onEdgesChange,
    onConnect,
    setNodes,
    setEdges,
    addNode: addNodeToStore,
    version: clientVersion,
  } = useFlowStore();
  const { undo, redo } = useTemporalStore((state) => ({
    undo: state.undo,
    redo: state.redo,
  }));
  const { sendJsonMessage, lastJsonMessage, mockServerState } = useMockSocket();
  const [contextMenu, setContextMenu] = useState<{
    x: number;
    y: number;
    nodeId?: string;
  } | null>(null);
  const [isFocusView, setFocusView] = useState(false);
  const [originalNodes, setOriginalNodes] = useState<AppNode[] | null>(null);
  const { toggleTheme } = useTheme();
  const [wsUrl, setWsUrl] = useState("ws://127.0.0.1:8000/ws (mocked)");
  const [isModalOpen, setIsModalOpen] = useState(false);

  const connectionStatus = "Connected (Mock)";

  const handleNodeDataChange = useCallback(
    (nodeId: string, data: Partial<NodeData>) => {
      setNodes(
        nodes.map((n) =>
          n.id === nodeId
            ? ({ ...n, data: { ...n.data, ...data } } as AppNode)
            : n,
        ),
      );
    },
    [nodes, setNodes],
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
    if (!lastJsonMessage) return;

    if (lastJsonMessage.type === "sync_graph") {
      const { graph, version } = lastJsonMessage.payload;
      if (lastJsonMessage.error === "version_mismatch") {
        alert(
          "Your graph is out of sync with the server. Your changes will be overwritten.",
        );
        setNodes(graph.nodes);
        setEdges(graph.edges);
        mockServerState.version = version;
      }
    } else if (lastJsonMessage.type === "apply_changes") {
      const { add = [] } = lastJsonMessage.payload;
      add.forEach((node: AppNode) => addNodeToStore(node));
      // In a real app, you would handle updates here
    }
  }, [lastJsonMessage, setNodes, setEdges, addNodeToStore, mockServerState]);

  const addNode = (
    type: "text" | "image" | "entity" | "component",
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
  };

  const onPaneContextMenu = useCallback(
    (event: React.MouseEvent | MouseEvent) => {
      event.preventDefault();
      const pane = (event.target as Element).closest(".react-flow__pane");
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
    setNodes(nodes.filter((n) => n.id !== contextMenu.nodeId));
    setEdges(
      edges.filter(
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

      setEdges([...edges, ...newEdges]);
    } catch (error) {
      alert(`Error fetching entity: ${error}`);
    }
  };

  useEffect(() => {
    sendJsonMessage({
      type: "sync_graph",
      payload: {
        version: clientVersion,
        graph: { nodes, edges },
      },
    });
  }, [nodes, edges, clientVersion, sendJsonMessage]);

  return (
    <div style={{ width: "100vw", height: "100vh" }}>
      {isFocusView && (
        <div style={{ position: "absolute", top: 10, left: 10, zIndex: 4 }}>
          <button onClick={exitFocusView}>Back to Global View</button>
          <button onClick={() => undo()}>Undo</button>
          <button onClick={() => redo()}>Redo</button>
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
          dynamicActions={
            contextMenu.nodeId &&
            nodes.find((n) => n.id === contextMenu.nodeId)?.type === "text"
              ? mockServerState.availableActions.text.map((action) => ({
                  ...action,
                  onClick: () =>
                    sendJsonMessage({
                      type: "execute_action",
                      payload: {
                        actionId: action.id,
                        nodeId: contextMenu.nodeId,
                      },
                    }),
                }))
              : []
          }
          onToggleTheme={toggleTheme}
          onAddTextNode={() =>
            addNode(
              "text",
              {
                label: "New Text Node",
                onChange: handleNodeDataChange,
                outputType: "text",
                inputType: "any",
              },
              { x: contextMenu.x, y: contextMenu.y },
            )
          }
          onAddImageNode={() =>
            addNode(
              "image",
              {
                url: "",
                onChange: handleNodeDataChange,
                outputType: "image",
                inputType: "any",
              },
              { x: contextMenu.x, y: contextMenu.y },
            )
          }
          isPaneMenu={!contextMenu.nodeId}
        />
      )}
      <StatusPanel
        status={connectionStatus}
        url={wsUrl}
        onClick={() => setIsModalOpen(true)}
      />
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
