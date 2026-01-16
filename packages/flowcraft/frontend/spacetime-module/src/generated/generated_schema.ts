/** AUTO-GENERATED - DO NOT EDIT **/
/* eslint-disable */
import { type Infer, t } from "spacetimedb/server";

export const Value = t.string();
export const Struct = t.string();
export const ListValue = t.string();

export const NodeKind = t.enum("NodeKind", {
  NODE_KIND_UNSPECIFIED: t.unit(),
  NODE_KIND_DYNAMIC: t.unit(),
  NODE_KIND_GROUP: t.unit(),
  NODE_KIND_PROCESS: t.unit(),
  NODE_KIND_NOTE: t.unit(),
});
export type NodeKind = Infer<typeof NodeKind>;

export const MutationSource = t.enum("MutationSource", {
  SOURCE_UNSPECIFIED: t.unit(),
  SOURCE_USER: t.unit(),
  SOURCE_REMOTE_TASK: t.unit(),
  SOURCE_SYSTEM: t.unit(),
  SOURCE_SYNC: t.unit(),
});
export type MutationSource = Infer<typeof MutationSource>;

export const PortMainType = t.enum("PortMainType", {
  PORT_MAIN_TYPE_UNSPECIFIED: t.unit(),
  PORT_MAIN_TYPE_ANY: t.unit(),
  PORT_MAIN_TYPE_STRING: t.unit(),
  PORT_MAIN_TYPE_NUMBER: t.unit(),
  PORT_MAIN_TYPE_BOOLEAN: t.unit(),
  PORT_MAIN_TYPE_IMAGE: t.unit(),
  PORT_MAIN_TYPE_VIDEO: t.unit(),
  PORT_MAIN_TYPE_AUDIO: t.unit(),
  PORT_MAIN_TYPE_LIST: t.unit(),
  PORT_MAIN_TYPE_SET: t.unit(),
  PORT_MAIN_TYPE_SYSTEM: t.unit(),
});
export type PortMainType = Infer<typeof PortMainType>;

export const MediaType = t.enum("MediaType", {
  MEDIA_UNSPECIFIED: t.unit(),
  MEDIA_IMAGE: t.unit(),
  MEDIA_VIDEO: t.unit(),
  MEDIA_AUDIO: t.unit(),
  MEDIA_MARKDOWN: t.unit(),
});
export type MediaType = Infer<typeof MediaType>;

export const VisualHint_Placement = t.enum("VisualHint_Placement", {
  UNSPECIFIED: t.unit(),
  CENTER_OF_VIEW: t.unit(),
  NEAR_SOURCE_NODE: t.unit(),
  BELOW_SELECTED_NODES: t.unit(),
  MOUSE_CURSOR: t.unit(),
});
export type VisualHint_Placement = Infer<typeof VisualHint_Placement>;

export const ActionExecutionStrategy = t.enum("ActionExecutionStrategy", {
  EXECUTION_IMMEDIATE: t.unit(),
  EXECUTION_BACKGROUND: t.unit(),
  EXECUTION_STREAMING: t.unit(),
});
export type ActionExecutionStrategy = Infer<typeof ActionExecutionStrategy>;

export const RenderMode = t.enum("RenderMode", {
  MODE_UNSPECIFIED: t.unit(),
  MODE_MEDIA: t.unit(),
  MODE_WIDGETS: t.unit(),
  MODE_MARKDOWN: t.unit(),
  MODE_CHAT: t.unit(),
});
export type RenderMode = Infer<typeof RenderMode>;

export const PortStyle = t.enum("PortStyle", {
  CIRCLE: t.unit(),
  SQUARE: t.unit(),
  DIAMOND: t.unit(),
  DASH: t.unit(),
});
export type PortStyle = Infer<typeof PortStyle>;

export const WidgetType = t.enum("WidgetType", {
  WIDGET_UNSPECIFIED: t.unit(),
  WIDGET_TEXT: t.unit(),
  WIDGET_SELECT: t.unit(),
  WIDGET_CHECKBOX: t.unit(),
  WIDGET_SLIDER: t.unit(),
  WIDGET_BUTTON: t.unit(),
});
export type WidgetType = Infer<typeof WidgetType>;

export const TaskStatus = t.enum("TaskStatus", {
  TASK_PENDING: t.unit(),
  TASK_PROCESSING: t.unit(),
  TASK_COMPLETED: t.unit(),
  TASK_FAILED: t.unit(),
  TASK_CANCELLED: t.unit(),
  TASK_RESTARTING: t.unit(),
});
export type TaskStatus = Infer<typeof TaskStatus>;

export const LogLevel = t.enum("LogLevel", {
  INFO: t.unit(),
  WARN: t.unit(),
  ERROR: t.unit(),
  DEBUG: t.unit(),
});
export type LogLevel = Infer<typeof LogLevel>;

export const PathUpdate_UpdateType = t.enum("PathUpdate_UpdateType", {
  REPLACE: t.unit(),
  MERGE: t.unit(),
  DELETE: t.unit(),
});
export type PathUpdate_UpdateType = Infer<typeof PathUpdate_UpdateType>;

// --- Position ---
export const Position = t.object("Position", {
  x: t.f64(),
  y: t.f64(),
});
export type Position = Infer<typeof Position>;

// --- Presentation ---
export const Presentation = t.object("Presentation", {
  position: t.option(Position),
  width: t.f64(),
  height: t.f64(),
  parentId: t.string(),
  zIndex: t.i32(),
  isHidden: t.bool(),
  isLocked: t.bool(),
  isInitialized: t.bool(),
});
export type Presentation = Infer<typeof Presentation>;

// --- VisualHint ---
export const VisualHint = t.object("VisualHint", {
  placement: t.option(VisualHint_Placement),
  referenceNodeId: t.string(),
  offset: t.option(Position),
});
export type VisualHint = Infer<typeof VisualHint>;

// --- Rect ---
export const Rect = t.object("Rect", {
  x: t.f64(),
  y: t.f64(),
  width: t.f64(),
  height: t.f64(),
});
export type Rect = Infer<typeof Rect>;

// --- Viewport ---
export const Viewport = t.object("Viewport", {
  x: t.f64(),
  y: t.f64(),
  zoom: t.f64(),
});
export type Viewport = Infer<typeof Viewport>;

// --- MediaContent ---
export const MediaContent = t.object("MediaContent", {
  type: t.option(MediaType),
  url: t.string(),
  content: t.string(),
  aspectRatio: t.f64(),
  galleryUrls: t.array(t.string()),
});
export type MediaContent = Infer<typeof MediaContent>;

// --- ChatMessagePart ---
export const ChatMessagePart_part = t.enum("ChatMessagePart_part", {
  text: t.string(),
  media: MediaContent,
});
export type ChatMessagePart_part = Infer<typeof ChatMessagePart_part>;

export const ChatMessagePart = t.object("ChatMessagePart", {
  part: t.option(ChatMessagePart_part),
    });
export type ChatMessagePart = Infer<typeof ChatMessagePart>;

// --- ChatMessagePreview ---
export const ChatMessagePreview = t.object("ChatMessagePreview", {
  role: t.string(),
  parts: t.array(t.option(ChatMessagePart)),
});
export type ChatMessagePreview = Infer<typeof ChatMessagePreview>;

// --- ChatActionParams ---
export const ChatActionParams = t.object("ChatActionParams", {
  userContent: t.string(),
  modelId: t.string(),
  endpointId: t.string(),
  useWebSearch: t.bool(),
  historyOverride: t.array(t.option(ChatMessagePreview)),
});
export type ChatActionParams = Infer<typeof ChatActionParams>;

// --- ChatEditParams ---
export const ChatEditParams = t.object("ChatEditParams", {
  messageId: t.string(),
  newParts: t.array(t.option(ChatMessagePart)),
});
export type ChatEditParams = Infer<typeof ChatEditParams>;

// --- ChatSwitchBranchParams ---
export const ChatSwitchBranchParams = t.object("ChatSwitchBranchParams", {
  targetMessageId: t.string(),
});
export type ChatSwitchBranchParams = Infer<typeof ChatSwitchBranchParams>;

// --- ChatSyncMessage ---
export const ChatSyncMessage = t.object("ChatSyncMessage", {
  id: t.string(),
  role: t.string(),
  parts: t.array(t.option(ChatMessagePart)),
  modelId: t.string(),
  timestamp: t.i64(),
});
export type ChatSyncMessage = Infer<typeof ChatSyncMessage>;

// --- ChatSyncBranchParams ---
export const ChatSyncBranchParams = t.object("ChatSyncBranchParams", {
  treeId: t.string(),
  anchorMessageId: t.string(),
  newMessages: t.array(t.option(ChatSyncMessage)),
  idempotencyKey: t.string(),
});
export type ChatSyncBranchParams = Infer<typeof ChatSyncBranchParams>;

// --- ImageEnhanceParams ---
export const ImageEnhanceParams = t.object("ImageEnhanceParams", {
  strength: t.f32(),
  modelName: t.string(),
});
export type ImageEnhanceParams = Infer<typeof ImageEnhanceParams>;

// --- NodeTransformParams ---
export const NodeTransformParams = t.object("NodeTransformParams", {
  instruction: t.string(),
  style: t.string(),
});
export type NodeTransformParams = Infer<typeof NodeTransformParams>;

// --- PromptGenParams ---
export const PromptGenParams = t.object("PromptGenParams", {
  prompt: t.string(),
  negativePrompt: t.string(),
  steps: t.i32(),
  cfgScale: t.f32(),
});
export type PromptGenParams = Infer<typeof PromptGenParams>;

// --- ActionTemplate ---
export const ActionTemplate = t.object("ActionTemplate", {
  id: t.string(),
  label: t.string(),
  path: t.array(t.string()),
  strategy: t.option(ActionExecutionStrategy),
  paramsSchema: t.option(Struct),
});
export type ActionTemplate = Infer<typeof ActionTemplate>;

// --- ActionDiscoveryRequest ---
export const ActionDiscoveryRequest = t.object("ActionDiscoveryRequest", {
  nodeId: t.string(),
  selectedNodeIds: t.array(t.string()),
});
export type ActionDiscoveryRequest = Infer<typeof ActionDiscoveryRequest>;

// --- ActionDiscoveryResponse ---
export const ActionDiscoveryResponse = t.object("ActionDiscoveryResponse", {
  actions: t.array(t.option(ActionTemplate)),
});
export type ActionDiscoveryResponse = Infer<typeof ActionDiscoveryResponse>;

// --- ActionExecutionRequest ---
export const ActionExecutionRequest_params = t.enum("ActionExecutionRequest_params", {
  paramsStruct: Struct,
  promptGen: PromptGenParams,
  transform: NodeTransformParams,
  enhance: ImageEnhanceParams,
  chatGenerate: ChatActionParams,
  chatSync: ChatSyncBranchParams,
});
export type ActionExecutionRequest_params = Infer<typeof ActionExecutionRequest_params>;

export const ActionExecutionRequest = t.object("ActionExecutionRequest", {
  actionId: t.string(),
  sourceNodeId: t.string(),
  contextNodeIds: t.array(t.string()),
  params: t.option(ActionExecutionRequest_params),
    });
export type ActionExecutionRequest = Infer<typeof ActionExecutionRequest>;

// --- ChatNodeState ---
export const ChatNodeState = t.object("ChatNodeState", {
  treeId: t.string(),
  conversationHeadId: t.string(),
  isHistoryCleared: t.bool(),
});
export type ChatNodeState = Infer<typeof ChatNodeState>;

// --- AiGenNodeState ---
export const AiGenNodeState = t.object("AiGenNodeState", {
  modelId: t.string(),
  progress: t.f32(),
  currentStatus: t.string(),
});
export type AiGenNodeState = Infer<typeof AiGenNodeState>;

// --- VisualNodeState ---
export const VisualNodeState = t.object("VisualNodeState", {
  url: t.string(),
  mimeType: t.string(),
  altText: t.string(),
});
export type VisualNodeState = Infer<typeof VisualNodeState>;

// --- DocumentNodeState ---
export const DocumentNodeState = t.object("DocumentNodeState", {
  content: t.string(),
  mimeType: t.string(),
});
export type DocumentNodeState = Infer<typeof DocumentNodeState>;

// --- AcousticNodeState ---
export const AcousticNodeState = t.object("AcousticNodeState", {
  url: t.string(),
  duration: t.f32(),
});
export type AcousticNodeState = Infer<typeof AcousticNodeState>;

// --- WidgetConfig ---
export const WidgetConfig = t.object("WidgetConfig", {
  placeholder: t.string(),
  min: t.f64(),
  max: t.f64(),
  step: t.f64(),
  dynamicOptions: t.bool(),
  actionTarget: t.string(),
});
export type WidgetConfig = Infer<typeof WidgetConfig>;

// --- WidgetOption ---
export const WidgetOption = t.object("WidgetOption", {
  label: t.string(),
  value: t.string(),
  description: t.string(),
});
export type WidgetOption = Infer<typeof WidgetOption>;

// --- Widget ---
export const Widget = t.object("Widget", {
  id: t.string(),
  type: t.option(WidgetType),
  label: t.string(),
  value: t.option(Value),
  config: t.option(WidgetConfig),
  options: t.array(t.option(WidgetOption)),
  isReadonly: t.bool(),
  isLoading: t.bool(),
  inputPortId: t.string(),
});
export type Widget = Infer<typeof Widget>;

// --- PortType ---
export const PortType = t.object("PortType", {
  mainType: t.option(PortMainType),
  itemType: t.string(),
  isGeneric: t.bool(),
});
export type PortType = Infer<typeof PortType>;

// --- Port ---
export const Port = t.object("Port", {
  id: t.string(),
  label: t.string(),
  type: t.option(PortType),
  color: t.string(),
  style: t.option(PortStyle),
  description: t.string(),
});
export type Port = Infer<typeof Port>;

// --- NodeData_MetadataEntry ---
export const NodeData_MetadataEntry = t.object("NodeData_MetadataEntry", {
  key: t.string(),
  value: t.string(),
});
export type NodeData_MetadataEntry = Infer<typeof NodeData_MetadataEntry>;

// --- NodeData ---
export const NodeData_extension = t.enum("NodeData_extension", {
  chat: ChatNodeState,
  aiGen: AiGenNodeState,
  visual: VisualNodeState,
  document: DocumentNodeState,
  acoustic: AcousticNodeState,
});
export type NodeData_extension = Infer<typeof NodeData_extension>;

export const NodeData = t.object("NodeData", {
  displayName: t.string(),
  availableModes: t.array(t.option(RenderMode)),
  activeMode: t.option(RenderMode),
  media: t.option(MediaContent),
  widgets: t.array(t.option(Widget)),
  inputPorts: t.array(t.option(Port)),
  outputPorts: t.array(t.option(Port)),
  metadata: t.array(t.option(NodeData_MetadataEntry)),
  taskId: t.string(),
  widgetsValues: t.option(Struct),
  widgetsSchema: t.option(Struct),
  extension: t.option(NodeData_extension),
    });
export type NodeData = Infer<typeof NodeData>;

// --- Node ---
export const Node = t.object("Node", {
  nodeId: t.string(),
  templateId: t.string(),
  nodeKind: t.option(NodeKind),
  presentation: t.option(Presentation),
  state: t.option(NodeData),
  visualHint: t.option(VisualHint),
  isSelected: t.bool(),
});
export type Node = Infer<typeof Node>;

// --- NodeTemplate ---
export const NodeTemplate = t.object("NodeTemplate", {
  templateId: t.string(),
  displayName: t.string(),
  menuPath: t.array(t.string()),
  defaultState: t.option(NodeData),
  defaultWidth: t.i32(),
  defaultHeight: t.i32(),
  widgetsSchema: t.option(Struct),
});
export type NodeTemplate = Infer<typeof NodeTemplate>;

// --- Edge_MetadataEntry ---
export const Edge_MetadataEntry = t.object("Edge_MetadataEntry", {
  key: t.string(),
  value: t.string(),
});
export type Edge_MetadataEntry = Infer<typeof Edge_MetadataEntry>;

// --- Edge ---
export const Edge = t.object("Edge", {
  edgeId: t.string(),
  sourceNodeId: t.string(),
  targetNodeId: t.string(),
  sourceHandle: t.string(),
  targetHandle: t.string(),
  metadata: t.array(t.option(Edge_MetadataEntry)),
});
export type Edge = Infer<typeof Edge>;

// --- TaskUpdate ---
export const TaskUpdate = t.object("TaskUpdate", {
  taskId: t.string(),
  status: t.option(TaskStatus),
  progress: t.f64(),
  message: t.string(),
  result: t.option(Value),
  nodeId: t.string(),
  displayLabel: t.string(),
  type: t.string(),
});
export type TaskUpdate = Infer<typeof TaskUpdate>;

// --- WidgetSignal ---
export const WidgetSignal_payload = t.enum("WidgetSignal_payload", {
  dataJson: t.string(),
  data: t.byteArray(),
});
export type WidgetSignal_payload = Infer<typeof WidgetSignal_payload>;

export const WidgetSignal = t.object("WidgetSignal", {
  nodeId: t.string(),
  widgetId: t.string(),
  payload: t.option(WidgetSignal_payload),
    });
export type WidgetSignal = Infer<typeof WidgetSignal>;

// --- RestartInstance ---
export const RestartInstance = t.object("RestartInstance", {
});
export type RestartInstance = Infer<typeof RestartInstance>;

// --- NodeSignal ---
export const NodeSignal_payload = t.enum("NodeSignal_payload", {
  parameters: Struct,
  chatGenerate: ChatActionParams,
  chatSync: ChatSyncBranchParams,
  chatEdit: ChatEditParams,
  chatSwitch: ChatSwitchBranchParams,
  restartInstance: RestartInstance,
});
export type NodeSignal_payload = Infer<typeof NodeSignal_payload>;

export const NodeSignal = t.object("NodeSignal", {
  nodeId: t.string(),
  payload: t.option(NodeSignal_payload),
    });
export type NodeSignal = Infer<typeof NodeSignal>;

// --- NodeIdList ---
export const NodeIdList = t.object("NodeIdList", {
  ids: t.array(t.string()),
});
export type NodeIdList = Infer<typeof NodeIdList>;

// --- HierarchyFilter ---
export const HierarchyFilter = t.object("HierarchyFilter", {
  rootNodeId: t.string(),
  depth: t.i32(),
});
export type HierarchyFilter = Infer<typeof HierarchyFilter>;

// --- SyncRequest ---
export const SyncRequest_filter = t.enum("SyncRequest_filter", {
  targetArea: Rect,
  nodeIds: NodeIdList,
  hierarchy: HierarchyFilter,
});
export type SyncRequest_filter = Infer<typeof SyncRequest_filter>;

export const SyncRequest = t.object("SyncRequest", {
  graphId: t.string(),
  subscribeToUpdates: t.bool(),
  filter: t.option(SyncRequest_filter),
    });
export type SyncRequest = Infer<typeof SyncRequest>;

// --- UpdateNodeRequest ---
export const UpdateNodeRequest = t.object("UpdateNodeRequest", {
  nodeId: t.string(),
  data: t.option(NodeData),
  presentation: t.option(Presentation),
});
export type UpdateNodeRequest = Infer<typeof UpdateNodeRequest>;

// --- UpdateWidgetRequest ---
export const UpdateWidgetRequest = t.object("UpdateWidgetRequest", {
  nodeId: t.string(),
  widgetId: t.string(),
  value: t.option(Value),
});
export type UpdateWidgetRequest = Infer<typeof UpdateWidgetRequest>;

// --- TaskCancelRequest ---
export const TaskCancelRequest = t.object("TaskCancelRequest", {
  taskId: t.string(),
});
export type TaskCancelRequest = Infer<typeof TaskCancelRequest>;

// --- ViewportUpdate ---
export const ViewportUpdate = t.object("ViewportUpdate", {
  viewport: t.option(Viewport),
  visibleBounds: t.option(Rect),
});
export type ViewportUpdate = Infer<typeof ViewportUpdate>;

// --- TemplateDiscoveryRequest ---
export const TemplateDiscoveryRequest = t.object("TemplateDiscoveryRequest", {
});
export type TemplateDiscoveryRequest = Infer<typeof TemplateDiscoveryRequest>;

// --- InferenceConfigDiscoveryRequest ---
export const InferenceConfigDiscoveryRequest = t.object("InferenceConfigDiscoveryRequest", {
});
export type InferenceConfigDiscoveryRequest = Infer<typeof InferenceConfigDiscoveryRequest>;

// --- GraphSnapshot ---
export const GraphSnapshot = t.object("GraphSnapshot", {
  nodes: t.array(t.option(Node)),
  edges: t.array(t.option(Edge)),
  version: t.i64(),
});
export type GraphSnapshot = Infer<typeof GraphSnapshot>;

// --- AddNode ---
export const AddNode = t.object("AddNode", {
  node: t.option(Node),
});
export type AddNode = Infer<typeof AddNode>;

// --- UpdateNode ---
export const UpdateNode = t.object("UpdateNode", {
  id: t.string(),
  data: t.option(NodeData),
  presentation: t.option(Presentation),
});
export type UpdateNode = Infer<typeof UpdateNode>;

// --- RemoveNode ---
export const RemoveNode = t.object("RemoveNode", {
  id: t.string(),
});
export type RemoveNode = Infer<typeof RemoveNode>;

// --- AddEdge ---
export const AddEdge = t.object("AddEdge", {
  edge: t.option(Edge),
});
export type AddEdge = Infer<typeof AddEdge>;

// --- RemoveEdge ---
export const RemoveEdge = t.object("RemoveEdge", {
  id: t.string(),
});
export type RemoveEdge = Infer<typeof RemoveEdge>;

// --- ClearGraph ---
export const ClearGraph = t.object("ClearGraph", {
});
export type ClearGraph = Infer<typeof ClearGraph>;

// --- PathUpdate ---
export const PathUpdate = t.object("PathUpdate", {
  targetId: t.string(),
  path: t.string(),
  value: t.option(Value),
  type: t.option(PathUpdate_UpdateType),
});
export type PathUpdate = Infer<typeof PathUpdate>;

// --- GraphMutation ---
export const GraphMutation_operation = t.enum("GraphMutation_operation", {
  addNode: AddNode,
  updateNode: UpdateNode,
  removeNode: RemoveNode,
  addEdge: AddEdge,
  removeEdge: RemoveEdge,
  clearGraph: ClearGraph,
  pathUpdate: PathUpdate,
});
export type GraphMutation_operation = Infer<typeof GraphMutation_operation>;

export const GraphMutation = t.object("GraphMutation", {
  originTaskId: t.string(),
  operation: t.option(GraphMutation_operation),
    });
export type GraphMutation = Infer<typeof GraphMutation>;

// --- MutationList ---
export const MutationList = t.object("MutationList", {
  mutations: t.array(t.option(GraphMutation)),
  sequenceNumber: t.i64(),
  source: t.option(MutationSource),
});
export type MutationList = Infer<typeof MutationList>;

// --- TemplateDiscoveryResponse ---
export const TemplateDiscoveryResponse = t.object("TemplateDiscoveryResponse", {
  templates: t.array(t.option(NodeTemplate)),
});
export type TemplateDiscoveryResponse = Infer<typeof TemplateDiscoveryResponse>;

// --- InferenceEndpointSummary ---
export const InferenceEndpointSummary = t.object("InferenceEndpointSummary", {
  id: t.string(),
  name: t.string(),
  models: t.array(t.string()),
});
export type InferenceEndpointSummary = Infer<typeof InferenceEndpointSummary>;

// --- InferenceConfigDiscoveryResponse ---
export const InferenceConfigDiscoveryResponse = t.object("InferenceConfigDiscoveryResponse", {
  endpoints: t.array(t.option(InferenceEndpointSummary)),
  defaultEndpointId: t.string(),
  defaultModel: t.string(),
});
export type InferenceConfigDiscoveryResponse = Infer<typeof InferenceConfigDiscoveryResponse>;

// --- ClearChatHistoryRequest ---
export const ClearChatHistoryRequest = t.object("ClearChatHistoryRequest", {
  nodeId: t.string(),
});
export type ClearChatHistoryRequest = Infer<typeof ClearChatHistoryRequest>;

// --- ChatStreamEvent ---
export const ChatStreamEvent = t.object("ChatStreamEvent", {
  chunkData: t.string(),
  isDone: t.bool(),
  messageId: t.string(),
});
export type ChatStreamEvent = Infer<typeof ChatStreamEvent>;

// --- LogEvent ---
export const LogEvent = t.object("LogEvent", {
  message: t.string(),
  level: t.option(LogLevel),
});
export type LogEvent = Infer<typeof LogEvent>;

// --- NodeProgress ---
export const NodeProgress = t.object("NodeProgress", {
  percentage: t.f32(),
  statusText: t.string(),
});
export type NodeProgress = Infer<typeof NodeProgress>;

// --- WidgetStreamEvent ---
export const WidgetStreamEvent = t.object("WidgetStreamEvent", {
  widgetId: t.string(),
  chunkData: t.string(),
  isDone: t.bool(),
});
export type WidgetStreamEvent = Infer<typeof WidgetStreamEvent>;

// --- NodeEvent ---
export const NodeEvent_payload = t.enum("NodeEvent_payload", {
  chatStream: ChatStreamEvent,
  log: LogEvent,
  progress: NodeProgress,
  widgetStream: WidgetStreamEvent,
  data: t.byteArray(),
});
export type NodeEvent_payload = Infer<typeof NodeEvent_payload>;

export const NodeEvent = t.object("NodeEvent", {
  nodeId: t.string(),
  payload: t.option(NodeEvent_payload),
    });
export type NodeEvent = Infer<typeof NodeEvent>;

// --- ErrorResponse ---
export const ErrorResponse = t.object("ErrorResponse", {
  code: t.string(),
  message: t.string(),
});
export type ErrorResponse = Infer<typeof ErrorResponse>;

// --- FlowMessage ---
export const FlowMessage_payload = t.enum("FlowMessage_payload", {
  syncRequest: SyncRequest,
  nodeUpdate: UpdateNodeRequest,
  widgetUpdate: UpdateWidgetRequest,
  actionExecute: ActionExecutionRequest,
  actionDiscovery: ActionDiscoveryRequest,
  taskCancel: TaskCancelRequest,
  viewportUpdate: ViewportUpdate,
  widgetSignal: WidgetSignal,
  nodeSignal: NodeSignal,
  templateDiscovery: TemplateDiscoveryRequest,
  inferenceDiscovery: InferenceConfigDiscoveryRequest,
  snapshot: GraphSnapshot,
  mutations: MutationList,
  actions: ActionDiscoveryResponse,
  templates: TemplateDiscoveryResponse,
  inferenceConfig: InferenceConfigDiscoveryResponse,
  taskUpdate: TaskUpdate,
  chatClear: ClearChatHistoryRequest,
  nodeEvent: NodeEvent,
  error: ErrorResponse,
});
export type FlowMessage_payload = Infer<typeof FlowMessage_payload>;

export const FlowMessage = t.object("FlowMessage", {
  messageId: t.string(),
  timestamp: t.i64(),
  payload: t.option(FlowMessage_payload),
    });
export type FlowMessage = Infer<typeof FlowMessage>;

// --- GetHistoryRequest ---
export const GetHistoryRequest = t.object("GetHistoryRequest", {
  graphId: t.string(),
  fromSeq: t.i64(),
  toSeq: t.i64(),
});
export type GetHistoryRequest = Infer<typeof GetHistoryRequest>;

// --- MutationLogEntry ---
export const MutationLogEntry = t.object("MutationLogEntry", {
  seq: t.i64(),
  mutation: t.option(GraphMutation),
  timestamp: t.i64(),
  source: t.option(MutationSource),
  description: t.string(),
  userId: t.string(),
});
export type MutationLogEntry = Infer<typeof MutationLogEntry>;

// --- HistoryResponse ---
export const HistoryResponse = t.object("HistoryResponse", {
  entries: t.array(t.option(MutationLogEntry)),
});
export type HistoryResponse = Infer<typeof HistoryResponse>;

// --- RollbackRequest ---
export const RollbackRequest = t.object("RollbackRequest", {
  graphId: t.string(),
  targetSeq: t.i64(),
});
export type RollbackRequest = Infer<typeof RollbackRequest>;

// --- GetChatHistoryRequest ---
export const GetChatHistoryRequest = t.object("GetChatHistoryRequest", {
  headId: t.string(),
});
export type GetChatHistoryRequest = Infer<typeof GetChatHistoryRequest>;

// --- ChatMsgMetadata ---
export const ChatMsgMetadata = t.object("ChatMsgMetadata", {
  modelId: t.string(),
  attachmentUrls: t.array(t.string()),
});
export type ChatMsgMetadata = Infer<typeof ChatMsgMetadata>;

// --- ChatMessage ---
export const ChatMessage_metadata = t.enum("ChatMessage_metadata", {
  metadataStruct: Struct,
  chatMetadata: ChatMsgMetadata,
});
export type ChatMessage_metadata = Infer<typeof ChatMessage_metadata>;

export const ChatMessage = t.object("ChatMessage", {
  id: t.string(),
  role: t.string(),
  parts: t.array(t.option(ChatMessagePart)),
  timestamp: t.i64(),
  parentId: t.string(),
  siblingIds: t.array(t.string()),
  treeId: t.string(),
  metadata: t.option(ChatMessage_metadata),
    });
export type ChatMessage = Infer<typeof ChatMessage>;

// --- ChatHistoryResponse ---
export const ChatHistoryResponse = t.object("ChatHistoryResponse", {
  entries: t.array(t.option(ChatMessage)),
});
export type ChatHistoryResponse = Infer<typeof ChatHistoryResponse>;

// --- AddSubGraph ---
export const AddSubGraph = t.object("AddSubGraph", {
  nodes: t.array(t.option(Node)),
  edges: t.array(t.option(Edge)),
});
export type AddSubGraph = Infer<typeof AddSubGraph>;


export const SCHEMA_METADATA = {
  messages: {
    "Position": Position,
    "Presentation": Presentation,
    "VisualHint": VisualHint,
    "Rect": Rect,
    "Viewport": Viewport,
    "MediaContent": MediaContent,
    "ChatMessagePart": ChatMessagePart,
    "ChatMessagePreview": ChatMessagePreview,
    "ChatActionParams": ChatActionParams,
    "ChatEditParams": ChatEditParams,
    "ChatSwitchBranchParams": ChatSwitchBranchParams,
    "ChatSyncMessage": ChatSyncMessage,
    "ChatSyncBranchParams": ChatSyncBranchParams,
    "ImageEnhanceParams": ImageEnhanceParams,
    "NodeTransformParams": NodeTransformParams,
    "PromptGenParams": PromptGenParams,
    "ActionTemplate": ActionTemplate,
    "ActionDiscoveryRequest": ActionDiscoveryRequest,
    "ActionDiscoveryResponse": ActionDiscoveryResponse,
    "ActionExecutionRequest": ActionExecutionRequest,
    "ChatNodeState": ChatNodeState,
    "AiGenNodeState": AiGenNodeState,
    "VisualNodeState": VisualNodeState,
    "DocumentNodeState": DocumentNodeState,
    "AcousticNodeState": AcousticNodeState,
    "WidgetConfig": WidgetConfig,
    "WidgetOption": WidgetOption,
    "Widget": Widget,
    "PortType": PortType,
    "Port": Port,
    "NodeData_MetadataEntry": NodeData_MetadataEntry,
    "NodeData": NodeData,
    "Node": Node,
    "NodeTemplate": NodeTemplate,
    "Edge_MetadataEntry": Edge_MetadataEntry,
    "Edge": Edge,
    "TaskUpdate": TaskUpdate,
    "WidgetSignal": WidgetSignal,
    "RestartInstance": RestartInstance,
    "NodeSignal": NodeSignal,
    "NodeIdList": NodeIdList,
    "HierarchyFilter": HierarchyFilter,
    "SyncRequest": SyncRequest,
    "UpdateNodeRequest": UpdateNodeRequest,
    "UpdateWidgetRequest": UpdateWidgetRequest,
    "TaskCancelRequest": TaskCancelRequest,
    "ViewportUpdate": ViewportUpdate,
    "TemplateDiscoveryRequest": TemplateDiscoveryRequest,
    "InferenceConfigDiscoveryRequest": InferenceConfigDiscoveryRequest,
    "GraphSnapshot": GraphSnapshot,
    "AddNode": AddNode,
    "UpdateNode": UpdateNode,
    "RemoveNode": RemoveNode,
    "AddEdge": AddEdge,
    "RemoveEdge": RemoveEdge,
    "ClearGraph": ClearGraph,
    "PathUpdate": PathUpdate,
    "GraphMutation": GraphMutation,
    "MutationList": MutationList,
    "TemplateDiscoveryResponse": TemplateDiscoveryResponse,
    "InferenceEndpointSummary": InferenceEndpointSummary,
    "InferenceConfigDiscoveryResponse": InferenceConfigDiscoveryResponse,
    "ClearChatHistoryRequest": ClearChatHistoryRequest,
    "ChatStreamEvent": ChatStreamEvent,
    "LogEvent": LogEvent,
    "NodeProgress": NodeProgress,
    "WidgetStreamEvent": WidgetStreamEvent,
    "NodeEvent": NodeEvent,
    "ErrorResponse": ErrorResponse,
    "FlowMessage": FlowMessage,
    "GetHistoryRequest": GetHistoryRequest,
    "MutationLogEntry": MutationLogEntry,
    "HistoryResponse": HistoryResponse,
    "RollbackRequest": RollbackRequest,
    "GetChatHistoryRequest": GetChatHistoryRequest,
    "ChatMsgMetadata": ChatMsgMetadata,
    "ChatMessage": ChatMessage,
    "ChatHistoryResponse": ChatHistoryResponse,
    "AddSubGraph": AddSubGraph,
  },
  enums: {
    "NodeKind": NodeKind,
    "MutationSource": MutationSource,
    "PortMainType": PortMainType,
    "MediaType": MediaType,
    "VisualHint_Placement": VisualHint_Placement,
    "ActionExecutionStrategy": ActionExecutionStrategy,
    "RenderMode": RenderMode,
    "PortStyle": PortStyle,
    "WidgetType": WidgetType,
    "TaskStatus": TaskStatus,
    "LogLevel": LogLevel,
    "PathUpdate_UpdateType": PathUpdate_UpdateType,
  }
} as const;
