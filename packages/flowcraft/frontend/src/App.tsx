import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  ReactFlow,
  MiniMap,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  addEdge,
  OnConnect,
  Node,
  applyNodeChanges,
} from 'reactflow';
import useWebSocket from 'react-use-websocket';
import { v4 as uuidv4 } from 'uuid';
import 'reactflow/dist/style.css';
import { TextNode } from './components/TextNode';
import { ImageNode } from './components/ImageNode';
import { ContextMenu } from './components/ContextMenu';
import { EntityNode } from './components/EntityNode';
import { ComponentNode } from './components/ComponentNode';

const WS_URL = 'ws://127.0.0.1:8000/ws';

function App() {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number; nodeId: string } | null>(null);
  const [isFocusView, setFocusView] = useState(false);
  const [originalNodes, setOriginalNodes] = useState<Node[] | null>(null);

  const { sendJsonMessage, lastJsonMessage } = useWebSocket(WS_URL, {
    share: true,
  });

  const handleNodeDataChange = (nodeId: string, data: any) => {
    const newNodes = nodes.map((n) =>
      n.id === nodeId ? { ...n, data: { ...n.data, ...data } } : n
    );
    setNodes(newNodes);
    sendJsonMessage({ nodes: newNodes, edges });
  };

  const nodeTypes = useMemo(() => ({
    text: (props: any) => <TextNode {...props} data={{ ...props.data, onChange: handleNodeDataChange }} />,
    image: (props: any) => <ImageNode {...props} data={{ ...props.data, onChange: handleNodeDataChange }} />,
    entity: EntityNode,
    component: ComponentNode,
  }), []);

  useEffect(() => {
    if (lastJsonMessage) {
      const graph = lastJsonMessage as { nodes: any[]; edges:any[] };
      setNodes(graph.nodes);
      setEdges(graph.edges);
    }
  }, [lastJsonMessage, setNodes, setEdges]);

  const onConnect: OnConnect = useCallback(
    (params) => {
      const newEdge = { ...params, id: `e${params.source}-${params.target}` };
      const newEdges = addEdge(newEdge, edges);
      setEdges(newEdges);
      sendJsonMessage({ nodes, edges: newEdges });
    },
    [edges, nodes, sendJsonMessage, setEdges]
  );

  const handleNodesChange = (changes: any) => {
    const updatedNodes = applyNodeChanges(changes, nodes);
    setNodes(updatedNodes);

    const dragChange = changes.find((change: any) => change.type === 'position' && !change.dragging);
    if (dragChange) {
      sendJsonMessage({ nodes: updatedNodes, edges });
    }
  };

  const addNode = (
    type: 'text' | 'image' | 'entity' | 'component',
    data: any,
    position: { x: number; y: number }
  ) => {
    const newNode = {
      id: uuidv4(),
      type,
      position,
      data,
    };
    const newNodes = [...nodes, newNode];
    setNodes(newNodes);
    return newNode;
  };

  const onNodeContextMenu = useCallback(
    (event: React.MouseEvent, node: Node) => {
      event.preventDefault();
      setContextMenu({ x: event.clientX, y: event.clientY, nodeId: node.id });
    },
    [setContextMenu]
  );

  const onPaneClick = useCallback(() => setContextMenu(null), [setContextMenu]);

  const handleDelete = () => {
    if (!contextMenu) return;
    const newNodes = nodes.filter((n) => n.id !== contextMenu.nodeId);
    const newEdges = edges.filter((e) => e.source !== contextMenu.nodeId && e.target !== contextMenu.nodeId);
    setNodes(newNodes);
    setEdges(newEdges);
    sendJsonMessage({ nodes: newNodes, edges: newEdges });
    setContextMenu(null);
  };

  const handleFocus = () => {
    if (!contextMenu) return;
    setOriginalNodes([...nodes]);
    const focusedNode = nodes.find((n) => n.id === contextMenu.nodeId);
    if (!focusedNode) return;

    const connectedEdges = edges.filter((e) => e.source === focusedNode.id || e.target === focusedNode.id);
    const neighborIds = new Set(connectedEdges.flatMap((e) => [e.source, e.target]));
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
      const components = await response.json();

      const entityNode = addNode('entity', { entityId }, { x: 250, y: 250 });

      const componentNodes = Object.entries(components).flatMap(([name, comps]: [string, any[]], i) =>
        comps.map((comp: any, j: number) => {
          const yOffset = (Object.keys(components).length / 2 - i) * 150;
          return addNode(
            'component',
            { componentName: name, json: JSON.stringify(comp, null, 2) },
            { x: 500, y: 250 + yOffset + j * 100 }
          );
        })
      );

      const newEdges = componentNodes.map(compNode => ({
        id: uuidv4(),
        source: entityNode.id,
        target: compNode.id,
      }));

      const newNodes = [...nodes, entityNode, ...componentNodes];
      const newEdgesWithConnections = [...edges, ...newEdges];
      setNodes(newNodes);
      setEdges(newEdgesWithConnections);
      sendJsonMessage({ nodes: newNodes, edges: newEdgesWithConnections });

    } catch (error) {
      alert(`Error fetching entity: ${error}`);
    }
  };

  return (
    <div style={{ width: '100vw', height: '100vh' }}>
      <div style={{ position: 'absolute', top: 10, left: 10, zIndex: 4 }}>
        {!isFocusView ? (
          <>
            <button
              onClick={() => addNode('text', { label: 'New Text Node' }, { x: Math.random() * 250, y: Math.random() * 250 })}
              style={{ marginRight: 5 }}
            >
              Add Text Node
            </button>
            <button
              onClick={() => addNode('image', { url: '' }, { x: Math.random() * 250, y: Math.random() * 250 })}
            >
              Add Image Node
            </button>
          </>
        ) : (
          <button onClick={exitFocusView}>Back to Global View</button>
        )}
      </div>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={handleNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeContextMenu={onNodeContextMenu}
        onPaneClick={onPaneClick}
        nodeTypes={nodeTypes}
        fitView
      >
        <Controls />
        <MiniMap />
        <Background variant="dots" gap={12} size={1} />
      </ReactFlow>
      {contextMenu && (
        <ContextMenu
          x={contextMenu.x}
          y={contextMenu.y}
          onClose={() => setContextMenu(null)}
          onDelete={handleDelete}
          onFocus={handleFocus}
          onShowDamEntity={onShowDamEntity}
        />
      )}
    </div>
  );
}

export default App;
