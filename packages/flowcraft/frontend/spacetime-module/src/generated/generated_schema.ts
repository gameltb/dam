/** AUTO-GENERATED - DO NOT EDIT **/
/* eslint-disable */
import { type Infer, t } from "spacetimedb/server";

export const Value = t.string();
export const Struct = t.string();
export const ListValue = t.string();

export const core_NodeKind = t.enum("core_NodeKind", {
  NODE_KIND_UNSPECIFIED: t.unit(),
  NODE_KIND_DYNAMIC: t.unit(),
  NODE_KIND_GROUP: t.unit(),
  NODE_KIND_PROCESS: t.unit(),
  NODE_KIND_NOTE: t.unit(),
});
export type core_NodeKind = Infer<typeof core_NodeKind>;

export const core_MutationSource = t.enum("core_MutationSource", {
  SOURCE_UNSPECIFIED: t.unit(),
  SOURCE_USER: t.unit(),
  SOURCE_REMOTE_TASK: t.unit(),
  SOURCE_SYSTEM: t.unit(),
  SOURCE_SYNC: t.unit(),
});
export type core_MutationSource = Infer<typeof core_MutationSource>;

export const core_PortMainType = t.enum("core_PortMainType", {
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
export type core_PortMainType = Infer<typeof core_PortMainType>;

export const core_MediaType = t.enum("core_MediaType", {
  MEDIA_UNSPECIFIED: t.unit(),
  MEDIA_IMAGE: t.unit(),
  MEDIA_VIDEO: t.unit(),
  MEDIA_AUDIO: t.unit(),
  MEDIA_MARKDOWN: t.unit(),
});
export type core_MediaType = Infer<typeof core_MediaType>;

export const core_VisualHint_Placement = t.enum("core_VisualHint_Placement", {
  UNSPECIFIED: t.unit(),
  CENTER_OF_VIEW: t.unit(),
  NEAR_SOURCE_NODE: t.unit(),
  BELOW_SELECTED_NODES: t.unit(),
  MOUSE_CURSOR: t.unit(),
});
export type core_VisualHint_Placement = Infer<typeof core_VisualHint_Placement>;

export const core_ActionExecutionStrategy = t.enum("core_ActionExecutionStrategy", {
  EXECUTION_IMMEDIATE: t.unit(),
  EXECUTION_BACKGROUND: t.unit(),
  EXECUTION_STREAMING: t.unit(),
});
export type core_ActionExecutionStrategy = Infer<typeof core_ActionExecutionStrategy>;

export const core_TaskStatus = t.enum("core_TaskStatus", {
  TASK_STATUS_PENDING: t.unit(),
  TASK_STATUS_CLAIMED: t.unit(),
  TASK_STATUS_RUNNING: t.unit(),
  TASK_STATUS_COMPLETED: t.unit(),
  TASK_STATUS_FAILED: t.unit(),
  TASK_STATUS_CANCELLED: t.unit(),
});
export type core_TaskStatus = Infer<typeof core_TaskStatus>;

export const core_WorkerLanguage = t.enum("core_WorkerLanguage", {
  WORKER_LANG_TS: t.unit(),
  WORKER_LANG_PYTHON: t.unit(),
  WORKER_LANG_RUST: t.unit(),
});
export type core_WorkerLanguage = Infer<typeof core_WorkerLanguage>;

export const core_RenderMode = t.enum("core_RenderMode", {
  MODE_UNSPECIFIED: t.unit(),
  MODE_MEDIA: t.unit(),
  MODE_WIDGETS: t.unit(),
  MODE_MARKDOWN: t.unit(),
  MODE_CHAT: t.unit(),
});
export type core_RenderMode = Infer<typeof core_RenderMode>;

export const core_PortStyle = t.enum("core_PortStyle", {
  CIRCLE: t.unit(),
  SQUARE: t.unit(),
  DIAMOND: t.unit(),
  DASH: t.unit(),
});
export type core_PortStyle = Infer<typeof core_PortStyle>;

export const core_WidgetType = t.enum("core_WidgetType", {
  WIDGET_UNSPECIFIED: t.unit(),
  WIDGET_TEXT: t.unit(),
  WIDGET_SELECT: t.unit(),
  WIDGET_CHECKBOX: t.unit(),
  WIDGET_SLIDER: t.unit(),
  WIDGET_BUTTON: t.unit(),
});
export type core_WidgetType = Infer<typeof core_WidgetType>;

export const services_LogLevel = t.enum("services_LogLevel", {
  INFO: t.unit(),
  WARN: t.unit(),
  ERROR: t.unit(),
  DEBUG: t.unit(),
});
export type services_LogLevel = Infer<typeof services_LogLevel>;

export const services_PathUpdateRequest_UpdateType = t.enum("services_PathUpdateRequest_UpdateType", {
  REPLACE: t.unit(),
  MERGE: t.unit(),
  DELETE: t.unit(),
});
export type services_PathUpdateRequest_UpdateType = Infer<typeof services_PathUpdateRequest_UpdateType>;

// --- core_MutationMetadata ---
export const core_MutationMetadata = t.object("core_MutationMetadata", {
  originTaskId: t.string(),
  timestamp: t.i64(),
  source: t.option(core_MutationSource),
  userId: t.string(),
});
export type core_MutationMetadata = Infer<typeof core_MutationMetadata>;

// --- core_Position ---
export const core_Position = t.object("core_Position", {
  x: t.f64(),
  y: t.f64(),
});
export type core_Position = Infer<typeof core_Position>;

// --- core_Presentation ---
export const core_Presentation = t.object("core_Presentation", {
  position: t.option(core_Position),
  width: t.f64(),
  height: t.f64(),
  parentId: t.string(),
  zIndex: t.i32(),
  isHidden: t.bool(),
  isLocked: t.bool(),
  isInitialized: t.bool(),
  isSelected: t.bool(),
});
export type core_Presentation = Infer<typeof core_Presentation>;

// --- core_VisualHint ---
export const core_VisualHint = t.object("core_VisualHint", {
  placement: t.option(core_VisualHint_Placement),
  referenceNodeId: t.string(),
  offset: t.option(core_Position),
});
export type core_VisualHint = Infer<typeof core_VisualHint>;

// --- core_Rect ---
export const core_Rect = t.object("core_Rect", {
  x: t.f64(),
  y: t.f64(),
  width: t.f64(),
  height: t.f64(),
});
export type core_Rect = Infer<typeof core_Rect>;

// --- core_Viewport ---
export const core_Viewport = t.object("core_Viewport", {
  x: t.f64(),
  y: t.f64(),
  zoom: t.f64(),
});
export type core_Viewport = Infer<typeof core_Viewport>;

// --- core_MediaContent ---
export const core_MediaContent = t.object("core_MediaContent", {
  type: t.option(core_MediaType),
  url: t.string(),
  content: t.string(),
  aspectRatio: t.f64(),
  galleryUrls: t.array(t.string()),
});
export type core_MediaContent = Infer<typeof core_MediaContent>;

// --- actions_ChatMessagePart ---
export const actions_ChatMessagePart_part = t.enum("actions_ChatMessagePart_part", {
  text: t.string(),
  media: core_MediaContent,
});
export type actions_ChatMessagePart_part = Infer<typeof actions_ChatMessagePart_part>;

export const actions_ChatMessagePart = t.object("actions_ChatMessagePart", {
  part: t.option(actions_ChatMessagePart_part),
    });
export type actions_ChatMessagePart = Infer<typeof actions_ChatMessagePart>;

// --- actions_ChatMessagePreview ---
export const actions_ChatMessagePreview = t.object("actions_ChatMessagePreview", {
  role: t.string(),
  parts: t.array(t.option(actions_ChatMessagePart)),
});
export type actions_ChatMessagePreview = Infer<typeof actions_ChatMessagePreview>;

// --- actions_ChatActionParams ---
export const actions_ChatActionParams = t.object("actions_ChatActionParams", {
  userContent: t.string(),
  modelId: t.string(),
  endpointId: t.string(),
  useWebSearch: t.bool(),
  historyOverride: t.array(t.option(actions_ChatMessagePreview)),
});
export type actions_ChatActionParams = Infer<typeof actions_ChatActionParams>;

// --- actions_ChatEditParams ---
export const actions_ChatEditParams = t.object("actions_ChatEditParams", {
  messageId: t.string(),
  newParts: t.array(t.option(actions_ChatMessagePart)),
});
export type actions_ChatEditParams = Infer<typeof actions_ChatEditParams>;

// --- actions_ChatSwitchBranchParams ---
export const actions_ChatSwitchBranchParams = t.object("actions_ChatSwitchBranchParams", {
  targetMessageId: t.string(),
});
export type actions_ChatSwitchBranchParams = Infer<typeof actions_ChatSwitchBranchParams>;

// --- actions_ChatSyncMessage ---
export const actions_ChatSyncMessage = t.object("actions_ChatSyncMessage", {
  id: t.string(),
  role: t.string(),
  parts: t.array(t.option(actions_ChatMessagePart)),
  modelId: t.string(),
  timestamp: t.i64(),
  parentId: t.string(),
});
export type actions_ChatSyncMessage = Infer<typeof actions_ChatSyncMessage>;

// --- actions_ChatSyncBranchParams ---
export const actions_ChatSyncBranchParams = t.object("actions_ChatSyncBranchParams", {
  treeId: t.string(),
  anchorMessageId: t.string(),
  newMessages: t.array(t.option(actions_ChatSyncMessage)),
  idempotencyKey: t.string(),
});
export type actions_ChatSyncBranchParams = Infer<typeof actions_ChatSyncBranchParams>;

// --- flowcraft_proto_v1_ImageEnhanceParams ---
export const flowcraft_proto_v1_ImageEnhanceParams = t.object("flowcraft_proto_v1_ImageEnhanceParams", {
  strength: t.f32(),
  modelName: t.string(),
});
export type flowcraft_proto_v1_ImageEnhanceParams = Infer<typeof flowcraft_proto_v1_ImageEnhanceParams>;

// --- flowcraft_proto_v1_NodeTransformParams ---
export const flowcraft_proto_v1_NodeTransformParams = t.object("flowcraft_proto_v1_NodeTransformParams", {
  instruction: t.string(),
  style: t.string(),
});
export type flowcraft_proto_v1_NodeTransformParams = Infer<typeof flowcraft_proto_v1_NodeTransformParams>;

// --- flowcraft_proto_v1_PromptGenParams ---
export const flowcraft_proto_v1_PromptGenParams = t.object("flowcraft_proto_v1_PromptGenParams", {
  prompt: t.string(),
  negativePrompt: t.string(),
  steps: t.i32(),
  cfgScale: t.f32(),
});
export type flowcraft_proto_v1_PromptGenParams = Infer<typeof flowcraft_proto_v1_PromptGenParams>;

// --- core_ActionTemplate ---
export const core_ActionTemplate = t.object("core_ActionTemplate", {
  id: t.string(),
  label: t.string(),
  menuPath: t.array(t.string()),
  strategy: t.option(core_ActionExecutionStrategy),
  paramsSchema: t.option(Struct),
});
export type core_ActionTemplate = Infer<typeof core_ActionTemplate>;

// --- core_ActionDiscoveryRequest ---
export const core_ActionDiscoveryRequest = t.object("core_ActionDiscoveryRequest", {
  nodeId: t.string(),
  selectedNodeIds: t.array(t.string()),
});
export type core_ActionDiscoveryRequest = Infer<typeof core_ActionDiscoveryRequest>;

// --- core_ActionDiscoveryResponse ---
export const core_ActionDiscoveryResponse = t.object("core_ActionDiscoveryResponse", {
  actions: t.array(t.option(core_ActionTemplate)),
});
export type core_ActionDiscoveryResponse = Infer<typeof core_ActionDiscoveryResponse>;

// --- core_PromptGenParams ---
export const core_PromptGenParams = t.object("core_PromptGenParams", {
  prompt: t.string(),
  negativePrompt: t.string(),
  steps: t.i32(),
  cfgScale: t.f32(),
});
export type core_PromptGenParams = Infer<typeof core_PromptGenParams>;

// --- core_NodeTransformParams ---
export const core_NodeTransformParams = t.object("core_NodeTransformParams", {
  instruction: t.string(),
  style: t.string(),
});
export type core_NodeTransformParams = Infer<typeof core_NodeTransformParams>;

// --- core_ImageEnhanceParams ---
export const core_ImageEnhanceParams = t.object("core_ImageEnhanceParams", {
  strength: t.f32(),
  modelName: t.string(),
});
export type core_ImageEnhanceParams = Infer<typeof core_ImageEnhanceParams>;

// --- core_ActionExecutionRequest ---
export const core_ActionExecutionRequest_params = t.enum("core_ActionExecutionRequest_params", {
  paramsStruct: Struct,
  promptGen: core_PromptGenParams,
  transform: core_NodeTransformParams,
  enhance: core_ImageEnhanceParams,
  chatGenerate: actions_ChatActionParams,
  chatSync: actions_ChatSyncBranchParams,
});
export type core_ActionExecutionRequest_params = Infer<typeof core_ActionExecutionRequest_params>;

export const core_ActionExecutionRequest = t.object("core_ActionExecutionRequest", {
  actionId: t.string(),
  sourceNodeId: t.string(),
  contextNodeIds: t.array(t.string()),
  params: t.option(core_ActionExecutionRequest_params),
    });
export type core_ActionExecutionRequest = Infer<typeof core_ActionExecutionRequest>;

// --- core_TaskUpdate ---
export const core_TaskUpdate = t.object("core_TaskUpdate", {
  taskId: t.string(),
  status: t.option(core_TaskStatus),
  progress: t.f64(),
  message: t.string(),
  result: t.option(Value),
  nodeId: t.string(),
  displayLabel: t.string(),
  type: t.string(),
});
export type core_TaskUpdate = Infer<typeof core_TaskUpdate>;

// --- core_NodeRuntimeState ---
export const core_NodeRuntimeState = t.object("core_NodeRuntimeState", {
  nodeId: t.string(),
  status: t.string(),
  progress: t.u32(),
  message: t.string(),
  error: t.string(),
  lastUpdated: t.u64(),
  activeUserId: t.string(),
});
export type core_NodeRuntimeState = Infer<typeof core_NodeRuntimeState>;

// --- core_WorkerSelector_MatchTagsEntry ---
export const core_WorkerSelector_MatchTagsEntry = t.object("core_WorkerSelector_MatchTagsEntry", {
  key: t.string(),
  value: t.string(),
});
export type core_WorkerSelector_MatchTagsEntry = Infer<typeof core_WorkerSelector_MatchTagsEntry>;

// --- core_WorkerSelector ---
export const core_WorkerSelector = t.object("core_WorkerSelector", {
  requiredCapability: t.string(),
  preferredWorkerId: t.string(),
  matchTags: t.array(t.option(core_WorkerSelector_MatchTagsEntry)),
});
export type core_WorkerSelector = Infer<typeof core_WorkerSelector>;

// --- core_TaskDefinition ---
export const core_TaskDefinition = t.object("core_TaskDefinition", {
  taskId: t.string(),
  nodeId: t.string(),
  taskType: t.string(),
  paramsPayload: t.byteArray(),
  selector: t.option(core_WorkerSelector),
  createdAt: t.i64(),
});
export type core_TaskDefinition = Infer<typeof core_TaskDefinition>;

// --- core_WorkerInfo_TagsEntry ---
export const core_WorkerInfo_TagsEntry = t.object("core_WorkerInfo_TagsEntry", {
  key: t.string(),
  value: t.string(),
});
export type core_WorkerInfo_TagsEntry = Infer<typeof core_WorkerInfo_TagsEntry>;

// --- core_WorkerInfo ---
export const core_WorkerInfo = t.object("core_WorkerInfo", {
  workerId: t.string(),
  lang: t.option(core_WorkerLanguage),
  capabilities: t.array(t.string()),
  tags: t.array(t.option(core_WorkerInfo_TagsEntry)),
  lastHeartbeat: t.i64(),
});
export type core_WorkerInfo = Infer<typeof core_WorkerInfo>;

// --- core_TaskAuditLog ---
export const core_TaskAuditLog = t.object("core_TaskAuditLog", {
  id: t.string(),
  taskId: t.string(),
  nodeId: t.string(),
  eventType: t.string(),
  message: t.string(),
  timestamp: t.i64(),
});
export type core_TaskAuditLog = Infer<typeof core_TaskAuditLog>;

// --- nodes_ChatNodeState ---
export const nodes_ChatNodeState = t.object("nodes_ChatNodeState", {
  treeId: t.string(),
  conversationHeadId: t.string(),
  isHistoryCleared: t.bool(),
});
export type nodes_ChatNodeState = Infer<typeof nodes_ChatNodeState>;

// --- nodes_AiGenNodeState ---
export const nodes_AiGenNodeState = t.object("nodes_AiGenNodeState", {
  modelId: t.string(),
  progress: t.f32(),
  currentStatus: t.string(),
});
export type nodes_AiGenNodeState = Infer<typeof nodes_AiGenNodeState>;

// --- nodes_VisualNodeState ---
export const nodes_VisualNodeState = t.object("nodes_VisualNodeState", {
  url: t.string(),
  mimeType: t.string(),
  altText: t.string(),
});
export type nodes_VisualNodeState = Infer<typeof nodes_VisualNodeState>;

// --- nodes_DocumentNodeState ---
export const nodes_DocumentNodeState = t.object("nodes_DocumentNodeState", {
  content: t.string(),
  mimeType: t.string(),
});
export type nodes_DocumentNodeState = Infer<typeof nodes_DocumentNodeState>;

// --- nodes_AcousticNodeState ---
export const nodes_AcousticNodeState = t.object("nodes_AcousticNodeState", {
  url: t.string(),
  duration: t.f32(),
});
export type nodes_AcousticNodeState = Infer<typeof nodes_AcousticNodeState>;

// --- core_WidgetConfig ---
export const core_WidgetConfig = t.object("core_WidgetConfig", {
  placeholder: t.string(),
  min: t.f64(),
  max: t.f64(),
  step: t.f64(),
  dynamicOptions: t.bool(),
  actionTarget: t.string(),
});
export type core_WidgetConfig = Infer<typeof core_WidgetConfig>;

// --- core_WidgetOption ---
export const core_WidgetOption = t.object("core_WidgetOption", {
  label: t.string(),
  value: t.string(),
  description: t.string(),
});
export type core_WidgetOption = Infer<typeof core_WidgetOption>;

// --- core_Widget ---
export const core_Widget = t.object("core_Widget", {
  id: t.string(),
  type: t.option(core_WidgetType),
  label: t.string(),
  value: t.option(Value),
  config: t.option(core_WidgetConfig),
  options: t.array(t.option(core_WidgetOption)),
  isReadonly: t.bool(),
  isLoading: t.bool(),
  inputPortId: t.string(),
});
export type core_Widget = Infer<typeof core_Widget>;

// --- core_PortType ---
export const core_PortType = t.object("core_PortType", {
  mainType: t.option(core_PortMainType),
  itemType: t.string(),
  isGeneric: t.bool(),
});
export type core_PortType = Infer<typeof core_PortType>;

// --- core_Port ---
export const core_Port = t.object("core_Port", {
  id: t.string(),
  label: t.string(),
  type: t.option(core_PortType),
  color: t.string(),
  style: t.option(core_PortStyle),
  description: t.string(),
});
export type core_Port = Infer<typeof core_Port>;

// --- core_NodeData_MetadataEntry ---
export const core_NodeData_MetadataEntry = t.object("core_NodeData_MetadataEntry", {
  key: t.string(),
  value: t.string(),
});
export type core_NodeData_MetadataEntry = Infer<typeof core_NodeData_MetadataEntry>;

// --- core_NodeData ---
export const core_NodeData_extension = t.enum("core_NodeData_extension", {
  chat: nodes_ChatNodeState,
  aiGen: nodes_AiGenNodeState,
  visual: nodes_VisualNodeState,
  document: nodes_DocumentNodeState,
  acoustic: nodes_AcousticNodeState,
});
export type core_NodeData_extension = Infer<typeof core_NodeData_extension>;

export const core_NodeData = t.object("core_NodeData", {
  displayName: t.string(),
  activeMode: t.option(core_RenderMode),
  taskId: t.string(),
  schemaVersion: t.i32(),
  availableModes: t.array(t.option(core_RenderMode)),
  media: t.option(core_MediaContent),
  widgets: t.array(t.option(core_Widget)),
  inputPorts: t.array(t.option(core_Port)),
  outputPorts: t.array(t.option(core_Port)),
  metadata: t.array(t.option(core_NodeData_MetadataEntry)),
  widgetsValues: t.option(Struct),
  widgetsSchema: t.option(Struct),
  extension: t.option(core_NodeData_extension),
    });
export type core_NodeData = Infer<typeof core_NodeData>;

// --- core_Node ---
export const core_Node = t.object("core_Node", {
  nodeId: t.string(),
  templateId: t.string(),
  nodeKind: t.option(core_NodeKind),
  presentation: t.option(core_Presentation),
  state: t.option(core_NodeData),
  visualHint: t.option(core_VisualHint),
});
export type core_Node = Infer<typeof core_Node>;

// --- core_NodeTemplate ---
export const core_NodeTemplate = t.object("core_NodeTemplate", {
  templateId: t.string(),
  displayName: t.string(),
  menuPath: t.array(t.string()),
  defaultState: t.option(core_NodeData),
  defaultWidth: t.i32(),
  defaultHeight: t.i32(),
  widgetsSchema: t.option(Struct),
});
export type core_NodeTemplate = Infer<typeof core_NodeTemplate>;

// --- core_Edge_MetadataEntry ---
export const core_Edge_MetadataEntry = t.object("core_Edge_MetadataEntry", {
  key: t.string(),
  value: t.string(),
});
export type core_Edge_MetadataEntry = Infer<typeof core_Edge_MetadataEntry>;

// --- core_Edge ---
export const core_Edge = t.object("core_Edge", {
  edgeId: t.string(),
  sourceNodeId: t.string(),
  targetNodeId: t.string(),
  sourceHandle: t.string(),
  targetHandle: t.string(),
  metadata: t.array(t.option(core_Edge_MetadataEntry)),
});
export type core_Edge = Infer<typeof core_Edge>;

// --- core_RestartInstance ---
export const core_RestartInstance = t.object("core_RestartInstance", {
});
export type core_RestartInstance = Infer<typeof core_RestartInstance>;

// --- core_WidgetSignal ---
export const core_WidgetSignal_payload = t.enum("core_WidgetSignal_payload", {
  dataJson: t.string(),
  data: t.byteArray(),
  parameters: Struct,
});
export type core_WidgetSignal_payload = Infer<typeof core_WidgetSignal_payload>;

export const core_WidgetSignal = t.object("core_WidgetSignal", {
  nodeId: t.string(),
  widgetId: t.string(),
  payload: t.option(core_WidgetSignal_payload),
    });
export type core_WidgetSignal = Infer<typeof core_WidgetSignal>;

// --- core_NodeSignal ---
export const core_NodeSignal_payload = t.enum("core_NodeSignal_payload", {
  parameters: Struct,
  chatGenerate: actions_ChatActionParams,
  chatSync: actions_ChatSyncBranchParams,
  chatEdit: actions_ChatEditParams,
  chatSwitch: actions_ChatSwitchBranchParams,
  restartInstance: core_RestartInstance,
  widgetSignal: core_WidgetSignal,
});
export type core_NodeSignal_payload = Infer<typeof core_NodeSignal_payload>;

export const core_NodeSignal = t.object("core_NodeSignal", {
  nodeId: t.string(),
  payload: t.option(core_NodeSignal_payload),
    });
export type core_NodeSignal = Infer<typeof core_NodeSignal>;

// --- services_NodeIdList ---
export const services_NodeIdList = t.object("services_NodeIdList", {
  ids: t.array(t.string()),
});
export type services_NodeIdList = Infer<typeof services_NodeIdList>;

// --- services_HierarchyFilter ---
export const services_HierarchyFilter = t.object("services_HierarchyFilter", {
  rootNodeId: t.string(),
  depth: t.i32(),
});
export type services_HierarchyFilter = Infer<typeof services_HierarchyFilter>;

// --- services_SyncRequest ---
export const services_SyncRequest_filter = t.enum("services_SyncRequest_filter", {
  targetArea: core_Rect,
  nodeIds: services_NodeIdList,
  hierarchy: services_HierarchyFilter,
});
export type services_SyncRequest_filter = Infer<typeof services_SyncRequest_filter>;

export const services_SyncRequest = t.object("services_SyncRequest", {
  graphId: t.string(),
  subscribeToUpdates: t.bool(),
  filter: t.option(services_SyncRequest_filter),
    });
export type services_SyncRequest = Infer<typeof services_SyncRequest>;

// --- services_TaskCancelRequest ---
export const services_TaskCancelRequest = t.object("services_TaskCancelRequest", {
  taskId: t.string(),
});
export type services_TaskCancelRequest = Infer<typeof services_TaskCancelRequest>;

// --- services_ViewportUpdate ---
export const services_ViewportUpdate = t.object("services_ViewportUpdate", {
  viewport: t.option(core_Viewport),
  visibleBounds: t.option(core_Rect),
});
export type services_ViewportUpdate = Infer<typeof services_ViewportUpdate>;

// --- services_TemplateDiscoveryRequest ---
export const services_TemplateDiscoveryRequest = t.object("services_TemplateDiscoveryRequest", {
});
export type services_TemplateDiscoveryRequest = Infer<typeof services_TemplateDiscoveryRequest>;

// --- services_InferenceConfigDiscoveryRequest ---
export const services_InferenceConfigDiscoveryRequest = t.object("services_InferenceConfigDiscoveryRequest", {
});
export type services_InferenceConfigDiscoveryRequest = Infer<typeof services_InferenceConfigDiscoveryRequest>;

// --- services_GraphSnapshot ---
export const services_GraphSnapshot = t.object("services_GraphSnapshot", {
  nodes: t.array(t.option(core_Node)),
  edges: t.array(t.option(core_Edge)),
  version: t.i64(),
});
export type services_GraphSnapshot = Infer<typeof services_GraphSnapshot>;

// --- services_AddNodeRequest ---
export const services_AddNodeRequest = t.object("services_AddNodeRequest", {
  metadata: t.option(core_MutationMetadata),
  node: t.option(core_Node),
});
export type services_AddNodeRequest = Infer<typeof services_AddNodeRequest>;

// --- services_RemoveNodeRequest ---
export const services_RemoveNodeRequest = t.object("services_RemoveNodeRequest", {
  metadata: t.option(core_MutationMetadata),
  id: t.string(),
});
export type services_RemoveNodeRequest = Infer<typeof services_RemoveNodeRequest>;

// --- services_AddEdgeRequest ---
export const services_AddEdgeRequest = t.object("services_AddEdgeRequest", {
  metadata: t.option(core_MutationMetadata),
  edge: t.option(core_Edge),
});
export type services_AddEdgeRequest = Infer<typeof services_AddEdgeRequest>;

// --- services_RemoveEdgeRequest ---
export const services_RemoveEdgeRequest = t.object("services_RemoveEdgeRequest", {
  metadata: t.option(core_MutationMetadata),
  id: t.string(),
});
export type services_RemoveEdgeRequest = Infer<typeof services_RemoveEdgeRequest>;

// --- services_ClearGraphRequest ---
export const services_ClearGraphRequest = t.object("services_ClearGraphRequest", {
  metadata: t.option(core_MutationMetadata),
});
export type services_ClearGraphRequest = Infer<typeof services_ClearGraphRequest>;

// --- services_PathUpdateRequest ---
export const services_PathUpdateRequest = t.object("services_PathUpdateRequest", {
  metadata: t.option(core_MutationMetadata),
  targetId: t.string(),
  path: t.string(),
  value: t.option(Value),
  type: t.option(services_PathUpdateRequest_UpdateType),
});
export type services_PathUpdateRequest = Infer<typeof services_PathUpdateRequest>;

// --- services_AddSubGraphRequest ---
export const services_AddSubGraphRequest = t.object("services_AddSubGraphRequest", {
  metadata: t.option(core_MutationMetadata),
  nodes: t.array(t.option(core_Node)),
  edges: t.array(t.option(core_Edge)),
});
export type services_AddSubGraphRequest = Infer<typeof services_AddSubGraphRequest>;

// --- services_ReparentNodeRequest ---
export const services_ReparentNodeRequest = t.object("services_ReparentNodeRequest", {
  metadata: t.option(core_MutationMetadata),
  nodeId: t.string(),
  newParentId: t.string(),
  newPosition: t.option(core_Position),
});
export type services_ReparentNodeRequest = Infer<typeof services_ReparentNodeRequest>;

// --- services_GraphMutation ---
export const services_GraphMutation_operation = t.enum("services_GraphMutation_operation", {
  addNode: services_AddNodeRequest,
  removeNode: services_RemoveNodeRequest,
  addEdge: services_AddEdgeRequest,
  removeEdge: services_RemoveEdgeRequest,
  clearGraph: services_ClearGraphRequest,
  pathUpdate: services_PathUpdateRequest,
  addSubGraph: services_AddSubGraphRequest,
  reparentNode: services_ReparentNodeRequest,
});
export type services_GraphMutation_operation = Infer<typeof services_GraphMutation_operation>;

export const services_GraphMutation = t.object("services_GraphMutation", {
  operation: t.option(services_GraphMutation_operation),
    });
export type services_GraphMutation = Infer<typeof services_GraphMutation>;

// --- services_MutationList ---
export const services_MutationList = t.object("services_MutationList", {
  mutations: t.array(t.option(services_GraphMutation)),
  sequenceNumber: t.i64(),
  source: t.option(core_MutationSource),
});
export type services_MutationList = Infer<typeof services_MutationList>;

// --- services_TemplateDiscoveryResponse ---
export const services_TemplateDiscoveryResponse = t.object("services_TemplateDiscoveryResponse", {
  templates: t.array(t.option(core_NodeTemplate)),
});
export type services_TemplateDiscoveryResponse = Infer<typeof services_TemplateDiscoveryResponse>;

// --- services_InferenceEndpointSummary ---
export const services_InferenceEndpointSummary = t.object("services_InferenceEndpointSummary", {
  id: t.string(),
  name: t.string(),
  models: t.array(t.string()),
});
export type services_InferenceEndpointSummary = Infer<typeof services_InferenceEndpointSummary>;

// --- services_InferenceConfigDiscoveryResponse ---
export const services_InferenceConfigDiscoveryResponse = t.object("services_InferenceConfigDiscoveryResponse", {
  endpoints: t.array(t.option(services_InferenceEndpointSummary)),
  defaultEndpointId: t.string(),
  defaultModel: t.string(),
});
export type services_InferenceConfigDiscoveryResponse = Infer<typeof services_InferenceConfigDiscoveryResponse>;

// --- services_ClearChatHistoryRequest ---
export const services_ClearChatHistoryRequest = t.object("services_ClearChatHistoryRequest", {
  nodeId: t.string(),
});
export type services_ClearChatHistoryRequest = Infer<typeof services_ClearChatHistoryRequest>;

// --- services_ResetNodeRequest ---
export const services_ResetNodeRequest = t.object("services_ResetNodeRequest", {
  nodeId: t.string(),
  clearData: t.bool(),
});
export type services_ResetNodeRequest = Infer<typeof services_ResetNodeRequest>;

// --- services_ChatStreamEvent ---
export const services_ChatStreamEvent = t.object("services_ChatStreamEvent", {
  chunkData: t.string(),
  isDone: t.bool(),
  messageId: t.string(),
});
export type services_ChatStreamEvent = Infer<typeof services_ChatStreamEvent>;

// --- services_LogEvent ---
export const services_LogEvent = t.object("services_LogEvent", {
  message: t.string(),
  level: t.option(services_LogLevel),
});
export type services_LogEvent = Infer<typeof services_LogEvent>;

// --- services_NodeProgress ---
export const services_NodeProgress = t.object("services_NodeProgress", {
  percentage: t.f32(),
  statusText: t.string(),
});
export type services_NodeProgress = Infer<typeof services_NodeProgress>;

// --- services_WidgetStreamEvent ---
export const services_WidgetStreamEvent = t.object("services_WidgetStreamEvent", {
  widgetId: t.string(),
  chunkData: t.string(),
  isDone: t.bool(),
});
export type services_WidgetStreamEvent = Infer<typeof services_WidgetStreamEvent>;

// --- services_NodeEvent ---
export const services_NodeEvent_payload = t.enum("services_NodeEvent_payload", {
  chatStream: services_ChatStreamEvent,
  log: services_LogEvent,
  progress: services_NodeProgress,
  widgetStream: services_WidgetStreamEvent,
  data: t.byteArray(),
});
export type services_NodeEvent_payload = Infer<typeof services_NodeEvent_payload>;

export const services_NodeEvent = t.object("services_NodeEvent", {
  nodeId: t.string(),
  payload: t.option(services_NodeEvent_payload),
    });
export type services_NodeEvent = Infer<typeof services_NodeEvent>;

// --- services_ErrorResponse ---
export const services_ErrorResponse = t.object("services_ErrorResponse", {
  code: t.string(),
  message: t.string(),
});
export type services_ErrorResponse = Infer<typeof services_ErrorResponse>;

// --- services_FlowMessage ---
export const services_FlowMessage_payload = t.enum("services_FlowMessage_payload", {
  syncRequest: services_SyncRequest,
  actionExecute: core_ActionExecutionRequest,
  actionDiscovery: core_ActionDiscoveryRequest,
  taskCancel: services_TaskCancelRequest,
  viewportUpdate: services_ViewportUpdate,
  widgetSignal: core_WidgetSignal,
  nodeSignal: core_NodeSignal,
  templateDiscovery: services_TemplateDiscoveryRequest,
  inferenceDiscovery: services_InferenceConfigDiscoveryRequest,
  snapshot: services_GraphSnapshot,
  mutations: services_MutationList,
  actions: core_ActionDiscoveryResponse,
  templates: services_TemplateDiscoveryResponse,
  inferenceConfig: services_InferenceConfigDiscoveryResponse,
  taskUpdate: core_TaskUpdate,
  chatClear: services_ClearChatHistoryRequest,
  resetNode: services_ResetNodeRequest,
  nodeEvent: services_NodeEvent,
  error: services_ErrorResponse,
});
export type services_FlowMessage_payload = Infer<typeof services_FlowMessage_payload>;

export const services_FlowMessage = t.object("services_FlowMessage", {
  messageId: t.string(),
  timestamp: t.i64(),
  payload: t.option(services_FlowMessage_payload),
    });
export type services_FlowMessage = Infer<typeof services_FlowMessage>;

// --- services_GetHistoryRequest ---
export const services_GetHistoryRequest = t.object("services_GetHistoryRequest", {
  graphId: t.string(),
  fromSeq: t.i64(),
  toSeq: t.i64(),
});
export type services_GetHistoryRequest = Infer<typeof services_GetHistoryRequest>;

// --- services_MutationLogEntry ---
export const services_MutationLogEntry = t.object("services_MutationLogEntry", {
  seq: t.i64(),
  mutation: t.option(services_GraphMutation),
  timestamp: t.i64(),
  source: t.option(core_MutationSource),
  description: t.string(),
  userId: t.string(),
});
export type services_MutationLogEntry = Infer<typeof services_MutationLogEntry>;

// --- services_HistoryResponse ---
export const services_HistoryResponse = t.object("services_HistoryResponse", {
  entries: t.array(t.option(services_MutationLogEntry)),
});
export type services_HistoryResponse = Infer<typeof services_HistoryResponse>;

// --- services_RollbackRequest ---
export const services_RollbackRequest = t.object("services_RollbackRequest", {
  graphId: t.string(),
  targetSeq: t.i64(),
});
export type services_RollbackRequest = Infer<typeof services_RollbackRequest>;

// --- services_GetChatHistoryRequest ---
export const services_GetChatHistoryRequest = t.object("services_GetChatHistoryRequest", {
  headId: t.string(),
});
export type services_GetChatHistoryRequest = Infer<typeof services_GetChatHistoryRequest>;

// --- services_ChatMsgMetadata ---
export const services_ChatMsgMetadata = t.object("services_ChatMsgMetadata", {
  modelId: t.string(),
  attachmentUrls: t.array(t.string()),
});
export type services_ChatMsgMetadata = Infer<typeof services_ChatMsgMetadata>;

// --- services_ChatMessage ---
export const services_ChatMessage_metadata = t.enum("services_ChatMessage_metadata", {
  metadataStruct: Struct,
  chatMetadata: services_ChatMsgMetadata,
});
export type services_ChatMessage_metadata = Infer<typeof services_ChatMessage_metadata>;

export const services_ChatMessage = t.object("services_ChatMessage", {
  id: t.string(),
  role: t.string(),
  parts: t.array(t.option(actions_ChatMessagePart)),
  timestamp: t.i64(),
  parentId: t.string(),
  siblingIds: t.array(t.string()),
  treeId: t.string(),
  metadata: t.option(services_ChatMessage_metadata),
    });
export type services_ChatMessage = Infer<typeof services_ChatMessage>;

// --- services_ChatHistoryResponse ---
export const services_ChatHistoryResponse = t.object("services_ChatHistoryResponse", {
  entries: t.array(t.option(services_ChatMessage)),
});
export type services_ChatHistoryResponse = Infer<typeof services_ChatHistoryResponse>;


export const SCHEMA_METADATA = {
  messages: {
    "core_MutationMetadata": core_MutationMetadata,
    "core_Position": core_Position,
    "core_Presentation": core_Presentation,
    "core_VisualHint": core_VisualHint,
    "core_Rect": core_Rect,
    "core_Viewport": core_Viewport,
    "core_MediaContent": core_MediaContent,
    "actions_ChatMessagePart": actions_ChatMessagePart,
    "actions_ChatMessagePreview": actions_ChatMessagePreview,
    "actions_ChatActionParams": actions_ChatActionParams,
    "actions_ChatEditParams": actions_ChatEditParams,
    "actions_ChatSwitchBranchParams": actions_ChatSwitchBranchParams,
    "actions_ChatSyncMessage": actions_ChatSyncMessage,
    "actions_ChatSyncBranchParams": actions_ChatSyncBranchParams,
    "flowcraft_proto_v1_ImageEnhanceParams": flowcraft_proto_v1_ImageEnhanceParams,
    "flowcraft_proto_v1_NodeTransformParams": flowcraft_proto_v1_NodeTransformParams,
    "flowcraft_proto_v1_PromptGenParams": flowcraft_proto_v1_PromptGenParams,
    "core_ActionTemplate": core_ActionTemplate,
    "core_ActionDiscoveryRequest": core_ActionDiscoveryRequest,
    "core_ActionDiscoveryResponse": core_ActionDiscoveryResponse,
    "core_PromptGenParams": core_PromptGenParams,
    "core_NodeTransformParams": core_NodeTransformParams,
    "core_ImageEnhanceParams": core_ImageEnhanceParams,
    "core_ActionExecutionRequest": core_ActionExecutionRequest,
    "core_TaskUpdate": core_TaskUpdate,
    "core_NodeRuntimeState": core_NodeRuntimeState,
    "core_WorkerSelector_MatchTagsEntry": core_WorkerSelector_MatchTagsEntry,
    "core_WorkerSelector": core_WorkerSelector,
    "core_TaskDefinition": core_TaskDefinition,
    "core_WorkerInfo_TagsEntry": core_WorkerInfo_TagsEntry,
    "core_WorkerInfo": core_WorkerInfo,
    "core_TaskAuditLog": core_TaskAuditLog,
    "nodes_ChatNodeState": nodes_ChatNodeState,
    "nodes_AiGenNodeState": nodes_AiGenNodeState,
    "nodes_VisualNodeState": nodes_VisualNodeState,
    "nodes_DocumentNodeState": nodes_DocumentNodeState,
    "nodes_AcousticNodeState": nodes_AcousticNodeState,
    "core_WidgetConfig": core_WidgetConfig,
    "core_WidgetOption": core_WidgetOption,
    "core_Widget": core_Widget,
    "core_PortType": core_PortType,
    "core_Port": core_Port,
    "core_NodeData_MetadataEntry": core_NodeData_MetadataEntry,
    "core_NodeData": core_NodeData,
    "core_Node": core_Node,
    "core_NodeTemplate": core_NodeTemplate,
    "core_Edge_MetadataEntry": core_Edge_MetadataEntry,
    "core_Edge": core_Edge,
    "core_RestartInstance": core_RestartInstance,
    "core_WidgetSignal": core_WidgetSignal,
    "core_NodeSignal": core_NodeSignal,
    "services_NodeIdList": services_NodeIdList,
    "services_HierarchyFilter": services_HierarchyFilter,
    "services_SyncRequest": services_SyncRequest,
    "services_TaskCancelRequest": services_TaskCancelRequest,
    "services_ViewportUpdate": services_ViewportUpdate,
    "services_TemplateDiscoveryRequest": services_TemplateDiscoveryRequest,
    "services_InferenceConfigDiscoveryRequest": services_InferenceConfigDiscoveryRequest,
    "services_GraphSnapshot": services_GraphSnapshot,
    "services_AddNodeRequest": services_AddNodeRequest,
    "services_RemoveNodeRequest": services_RemoveNodeRequest,
    "services_AddEdgeRequest": services_AddEdgeRequest,
    "services_RemoveEdgeRequest": services_RemoveEdgeRequest,
    "services_ClearGraphRequest": services_ClearGraphRequest,
    "services_PathUpdateRequest": services_PathUpdateRequest,
    "services_AddSubGraphRequest": services_AddSubGraphRequest,
    "services_ReparentNodeRequest": services_ReparentNodeRequest,
    "services_GraphMutation": services_GraphMutation,
    "services_MutationList": services_MutationList,
    "services_TemplateDiscoveryResponse": services_TemplateDiscoveryResponse,
    "services_InferenceEndpointSummary": services_InferenceEndpointSummary,
    "services_InferenceConfigDiscoveryResponse": services_InferenceConfigDiscoveryResponse,
    "services_ClearChatHistoryRequest": services_ClearChatHistoryRequest,
    "services_ResetNodeRequest": services_ResetNodeRequest,
    "services_ChatStreamEvent": services_ChatStreamEvent,
    "services_LogEvent": services_LogEvent,
    "services_NodeProgress": services_NodeProgress,
    "services_WidgetStreamEvent": services_WidgetStreamEvent,
    "services_NodeEvent": services_NodeEvent,
    "services_ErrorResponse": services_ErrorResponse,
    "services_FlowMessage": services_FlowMessage,
    "services_GetHistoryRequest": services_GetHistoryRequest,
    "services_MutationLogEntry": services_MutationLogEntry,
    "services_HistoryResponse": services_HistoryResponse,
    "services_RollbackRequest": services_RollbackRequest,
    "services_GetChatHistoryRequest": services_GetChatHistoryRequest,
    "services_ChatMsgMetadata": services_ChatMsgMetadata,
    "services_ChatMessage": services_ChatMessage,
    "services_ChatHistoryResponse": services_ChatHistoryResponse,
  },
  enums: {
    "core_NodeKind": core_NodeKind,
    "core_MutationSource": core_MutationSource,
    "core_PortMainType": core_PortMainType,
    "core_MediaType": core_MediaType,
    "core_VisualHint_Placement": core_VisualHint_Placement,
    "core_ActionExecutionStrategy": core_ActionExecutionStrategy,
    "core_TaskStatus": core_TaskStatus,
    "core_WorkerLanguage": core_WorkerLanguage,
    "core_RenderMode": core_RenderMode,
    "core_PortStyle": core_PortStyle,
    "core_WidgetType": core_WidgetType,
    "services_LogLevel": services_LogLevel,
    "services_PathUpdateRequest_UpdateType": services_PathUpdateRequest_UpdateType,
  }
} as const;
