import { CheckIcon, GlobeIcon } from "lucide-react";
import React, { useState } from "react";

import { type InferenceConfigDiscoveryResponse } from "@/generated/flowcraft/v1/core/service_pb";
import { useUiStore } from "@/store/uiStore";

import {
  ModelSelector,
  ModelSelectorContent,
  ModelSelectorGroup,
  ModelSelectorItem,
  ModelSelectorList,
  ModelSelectorName,
  ModelSelectorTrigger,
} from "../../ai-elements/model-selector";
import {
  PromptInput,
  PromptInputActionAddAttachments,
  PromptInputActionMenu,
  PromptInputActionMenuContent,
  PromptInputActionMenuTrigger,
  PromptInputAttachment,
  PromptInputAttachments,
  PromptInputBody,
  PromptInputButton,
  PromptInputFooter,
  type PromptInputMessage,
  PromptInputSubmit,
  PromptInputTextarea,
  PromptInputTools,
} from "../../ai-elements/prompt-input";
import { Suggestion, Suggestions } from "../../ai-elements/suggestion";
import { type ChatStatus, type ContextNode } from "./types";

const SUGGESTIONS = [
  "Explain how this graph works",
  "Summarize current results",
  "Optimize this workflow",
];

interface Props {
  droppedNodes: ContextNode[];
  inferenceConfig: InferenceConfigDiscoveryResponse | null;
  onModelChange: (m: string, endpoint?: string) => void;
  onSubmit: (
    msg: PromptInputMessage,
    model: string,
    endpoint: string,
    search: boolean,
  ) => void;
  onWebSearchChange: (v: boolean) => void;
  selectedEndpoint: string;
  selectedModel: string;
  setDroppedNodes: (n: ContextNode[]) => void;
  status: ChatStatus;
  useWebSearch: boolean;
}

export const ChatInputArea: React.FC<Props> = ({
  droppedNodes,
  inferenceConfig,
  onModelChange,
  onSubmit,
  onWebSearchChange,
  selectedEndpoint,
  selectedModel,
  setDroppedNodes,
  status,
  useWebSearch,
}) => {
  const [inputText, setInputText] = useState("");
  const [modelSelectorOpen, setModelSelectorOpen] = useState(false);
  const localClients = useUiStore((s) => s.settings.localClients);

  const allModels = React.useMemo(() => {
    const models: {
      endpointId: string;
      endpointName: string;
      id: string;
      isLocal?: boolean;
      name: string;
    }[] = [];

    // Add Server Models
    if (inferenceConfig) {
      inferenceConfig.endpoints.forEach((e) => {
        e.models.forEach((m) => {
          models.push({
            endpointId: e.id,
            endpointName: e.name,
            id: m,
            name: m,
          });
        });
      });
    }

    // Add Local Models
    localClients.forEach((c) => {
      models.push({
        endpointId: c.id,
        endpointName: `${c.name} (Local)`,
        id: c.model,
        isLocal: true,
        name: c.model,
      });
    });

    return models;
  }, [inferenceConfig, localClients]);

  const currentModel = allModels.find(
    (m) => m.id === selectedModel && m.endpointId === selectedEndpoint,
  );
  const currentModelName = currentModel?.name ?? selectedModel;

  const handleSubmit = (msg: PromptInputMessage) => {
    if (status !== "ready") return;
    onSubmit(msg, selectedModel, selectedEndpoint, useWebSearch);
    setInputText("");
  };

  return (
    <div className="shrink-0 bg-muted/5 border-t border-node-border">
      <div className="grid gap-2 pt-2">
        {droppedNodes.length > 0 && (
          <div className="flex flex-wrap gap-2 px-2">
            {droppedNodes.map((n) => (
              <div
                className="group flex items-center gap-1.5 bg-primary/10 text-primary px-2 py-1 rounded-md text-xs border border-primary/20"
                key={n.id}
              >
                <span>{n.label}</span>
                <button
                  className="hover:text-destructive"
                  onClick={() => {
                    setDroppedNodes(droppedNodes.filter((i) => i.id !== n.id));
                  }}
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        )}

        <Suggestions className="px-2">
          {SUGGESTIONS.map((s) => (
            <Suggestion
              key={s}
              onClick={() => {
                if (status !== "ready") return;
                onSubmit(
                  { files: [], text: s },
                  selectedModel,
                  selectedEndpoint,
                  useWebSearch,
                );
              }}
              suggestion={s}
            />
          ))}
        </Suggestions>

        <div className="px-2 pb-2">
          <PromptInput onSubmit={handleSubmit}>
            <PromptInputAttachments>
              {(f) => <PromptInputAttachment data={f} key={f.id} />}
            </PromptInputAttachments>
            <PromptInputBody>
              <PromptInputTextarea
                className="min-h-[44px]"
                disabled={status !== "ready"}
                onChange={(e) => {
                  setInputText(e.target.value);
                }}
                placeholder={
                  status === "ready" ? "Ask anything..." : "Please wait..."
                }
                value={inputText}
              />
            </PromptInputBody>
            <PromptInputFooter>
              <PromptInputTools>
                <PromptInputActionMenu>
                  <PromptInputActionMenuTrigger />
                  <PromptInputActionMenuContent>
                    <PromptInputActionAddAttachments label="Upload" />
                  </PromptInputActionMenuContent>
                </PromptInputActionMenu>
                <PromptInputButton
                  onClick={() => {
                    onWebSearchChange(!useWebSearch);
                  }}
                  variant={useWebSearch ? "default" : "ghost"}
                >
                  <GlobeIcon size={14} />
                </PromptInputButton>
                <ModelSelector
                  onOpenChange={setModelSelectorOpen}
                  open={modelSelectorOpen}
                >
                  <ModelSelectorTrigger asChild>
                    <PromptInputButton>
                      <span>{currentModelName}</span>
                    </PromptInputButton>
                  </ModelSelectorTrigger>
                  <ModelSelectorContent>
                    <ModelSelectorList>
                      {/* Server Endpoints */}
                      {inferenceConfig?.endpoints.map((e) => (
                        <ModelSelectorGroup heading={e.name} key={e.id}>
                          {e.models.map((m) => (
                            <ModelSelectorItem
                              key={`${e.id}-${m}`}
                              onSelect={() => {
                                onModelChange(m, e.id);
                                setModelSelectorOpen(false);
                              }}
                              value={m}
                            >
                              <ModelSelectorName>{m}</ModelSelectorName>
                              {selectedModel === m &&
                                selectedEndpoint === e.id && (
                                  <CheckIcon className="ml-auto size-3" />
                                )}
                            </ModelSelectorItem>
                          ))}
                        </ModelSelectorGroup>
                      ))}

                      {/* Local Clients */}
                      {localClients.length > 0 && (
                        <ModelSelectorGroup heading="Local Clients">
                          {localClients.map((c) => (
                            <ModelSelectorItem
                              key={c.id}
                              onSelect={() => {
                                onModelChange(c.model, c.id);
                                setModelSelectorOpen(false);
                              }}
                              value={c.id}
                            >
                              <ModelSelectorName>
                                {c.name} ({c.model})
                              </ModelSelectorName>
                              {selectedEndpoint === c.id && (
                                <CheckIcon className="ml-auto size-3" />
                              )}
                            </ModelSelectorItem>
                          ))}
                        </ModelSelectorGroup>
                      )}
                    </ModelSelectorList>
                  </ModelSelectorContent>
                </ModelSelector>
              </PromptInputTools>
              <PromptInputSubmit
                status={
                  status === "streaming"
                    ? "streaming"
                    : status === "submitted"
                      ? "submitted"
                      : "ready"
                }
              />
            </PromptInputFooter>
          </PromptInput>
        </div>
      </div>
    </div>
  );
};
