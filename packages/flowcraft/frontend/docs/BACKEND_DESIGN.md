# Backend Design Documentation

This document outlines the design of the mock backend for Flowcraft.

## Overview

The backend is simulated using a hybrid approach:

1.  **MSW (Mock Service Worker)**: For legacy REST endpoints (like `/api/node-templates`).
2.  **ConnectRPC (gRPC-web compatible)**: For the core Flowcraft service. This provides a type-safe, contract-first API using Protobuf.

The frontend communicates with this mock service via an in-memory `routerTransport`, which can be easily swapped for a real network transport (Connect, gRPC-web, or gRPC).

## State Management

SpacetimeDB serves as the primary Source of Truth for the application state:

- **Tables**: Managed via `spacetime-module/src/tables/`, using Protobuf-derived schemas.
- **Persistence**: Real-time synchronization between the Node.js worker and the React frontend.

## Protocol & Implementation

### 1. Direct Message Reducers

The backend implementation uses direct Reducers for individual operations, eliminating the need for complex multiplexing:

- **createNodePb**: Handles atomic node creation.
- **updateNodePb**: Handles differential updates to node state or presentation.
- **addEdgePb**: Manages connection establishment.
- **executeAction**: Dispatches tasks to the worker pool.

### 2. Protobuf Schema

All data structures are strictly defined in `schema/flowcraft/v1/` using hierarchical packages:

- `flowcraft.v1.core`: Fundamental types (Node, Edge, Port).
- `flowcraft.v1.services`: Service definitions and Request/Response messages.
- `flowcraft.v1.nodes`: Domain-specific node states (Chat, Media).

## Node Definitions (JSON/Proto)

Nodes are defined dynamically. A typical node definition includes:

- `modes`: List of supported rendering modes.
- `media`: Image/Video info + `aspectRatio` + `gallery` (optional array of strings).
- `widgets`: Array of widget definitions (type, label, value, options).
