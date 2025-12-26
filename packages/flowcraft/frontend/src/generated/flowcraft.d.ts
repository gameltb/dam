import * as $protobuf from "protobufjs";
import Long = require("long");
/** Namespace flowcraft. */
export namespace flowcraft {

    /** Namespace v1. */
    namespace v1 {

        /** ActionExecutionStrategy enum. */
        enum ActionExecutionStrategy {
            EXECUTION_IMMEDIATE = 0,
            EXECUTION_TASK = 1
        }

        /** Properties of an ActionTemplate. */
        interface IActionTemplate {

            /** ActionTemplate id */
            id?: (string|null);

            /** ActionTemplate label */
            label?: (string|null);

            /** ActionTemplate path */
            path?: (string[]|null);

            /** ActionTemplate strategy */
            strategy?: (flowcraft.v1.ActionExecutionStrategy|null);

            /** ActionTemplate description */
            description?: (string|null);

            /** ActionTemplate icon */
            icon?: (string|null);
        }

        /** Represents an ActionTemplate. */
        class ActionTemplate implements IActionTemplate {

            /**
             * Constructs a new ActionTemplate.
             * @param [properties] Properties to set
             */
            constructor(properties?: flowcraft.v1.IActionTemplate);

            /** ActionTemplate id. */
            public id: string;

            /** ActionTemplate label. */
            public label: string;

            /** ActionTemplate path. */
            public path: string[];

            /** ActionTemplate strategy. */
            public strategy: flowcraft.v1.ActionExecutionStrategy;

            /** ActionTemplate description. */
            public description: string;

            /** ActionTemplate icon. */
            public icon: string;

            /**
             * Creates a new ActionTemplate instance using the specified properties.
             * @param [properties] Properties to set
             * @returns ActionTemplate instance
             */
            public static create(properties?: flowcraft.v1.IActionTemplate): flowcraft.v1.ActionTemplate;

            /**
             * Encodes the specified ActionTemplate message. Does not implicitly {@link flowcraft.v1.ActionTemplate.verify|verify} messages.
             * @param message ActionTemplate message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: flowcraft.v1.IActionTemplate, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified ActionTemplate message, length delimited. Does not implicitly {@link flowcraft.v1.ActionTemplate.verify|verify} messages.
             * @param message ActionTemplate message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: flowcraft.v1.IActionTemplate, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes an ActionTemplate message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns ActionTemplate
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): flowcraft.v1.ActionTemplate;

            /**
             * Decodes an ActionTemplate message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns ActionTemplate
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): flowcraft.v1.ActionTemplate;

            /**
             * Verifies an ActionTemplate message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates an ActionTemplate message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns ActionTemplate
             */
            public static fromObject(object: { [k: string]: any }): flowcraft.v1.ActionTemplate;

            /**
             * Creates a plain object from an ActionTemplate message. Also converts values to other types if specified.
             * @param message ActionTemplate
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: flowcraft.v1.ActionTemplate, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this ActionTemplate to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };

            /**
             * Gets the default type url for ActionTemplate
             * @param [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns The default type url
             */
            public static getTypeUrl(typeUrlPrefix?: string): string;
        }

        /** Properties of an ActionDiscoveryRequest. */
        interface IActionDiscoveryRequest {

            /** ActionDiscoveryRequest nodeId */
            nodeId?: (string|null);

            /** ActionDiscoveryRequest selectedNodeIds */
            selectedNodeIds?: (string[]|null);
        }

        /** Represents an ActionDiscoveryRequest. */
        class ActionDiscoveryRequest implements IActionDiscoveryRequest {

            /**
             * Constructs a new ActionDiscoveryRequest.
             * @param [properties] Properties to set
             */
            constructor(properties?: flowcraft.v1.IActionDiscoveryRequest);

            /** ActionDiscoveryRequest nodeId. */
            public nodeId: string;

            /** ActionDiscoveryRequest selectedNodeIds. */
            public selectedNodeIds: string[];

            /**
             * Creates a new ActionDiscoveryRequest instance using the specified properties.
             * @param [properties] Properties to set
             * @returns ActionDiscoveryRequest instance
             */
            public static create(properties?: flowcraft.v1.IActionDiscoveryRequest): flowcraft.v1.ActionDiscoveryRequest;

            /**
             * Encodes the specified ActionDiscoveryRequest message. Does not implicitly {@link flowcraft.v1.ActionDiscoveryRequest.verify|verify} messages.
             * @param message ActionDiscoveryRequest message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: flowcraft.v1.IActionDiscoveryRequest, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified ActionDiscoveryRequest message, length delimited. Does not implicitly {@link flowcraft.v1.ActionDiscoveryRequest.verify|verify} messages.
             * @param message ActionDiscoveryRequest message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: flowcraft.v1.IActionDiscoveryRequest, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes an ActionDiscoveryRequest message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns ActionDiscoveryRequest
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): flowcraft.v1.ActionDiscoveryRequest;

            /**
             * Decodes an ActionDiscoveryRequest message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns ActionDiscoveryRequest
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): flowcraft.v1.ActionDiscoveryRequest;

            /**
             * Verifies an ActionDiscoveryRequest message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates an ActionDiscoveryRequest message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns ActionDiscoveryRequest
             */
            public static fromObject(object: { [k: string]: any }): flowcraft.v1.ActionDiscoveryRequest;

            /**
             * Creates a plain object from an ActionDiscoveryRequest message. Also converts values to other types if specified.
             * @param message ActionDiscoveryRequest
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: flowcraft.v1.ActionDiscoveryRequest, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this ActionDiscoveryRequest to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };

            /**
             * Gets the default type url for ActionDiscoveryRequest
             * @param [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns The default type url
             */
            public static getTypeUrl(typeUrlPrefix?: string): string;
        }

        /** Properties of an ActionDiscoveryResponse. */
        interface IActionDiscoveryResponse {

            /** ActionDiscoveryResponse actions */
            actions?: (flowcraft.v1.IActionTemplate[]|null);
        }

        /** Represents an ActionDiscoveryResponse. */
        class ActionDiscoveryResponse implements IActionDiscoveryResponse {

            /**
             * Constructs a new ActionDiscoveryResponse.
             * @param [properties] Properties to set
             */
            constructor(properties?: flowcraft.v1.IActionDiscoveryResponse);

            /** ActionDiscoveryResponse actions. */
            public actions: flowcraft.v1.IActionTemplate[];

            /**
             * Creates a new ActionDiscoveryResponse instance using the specified properties.
             * @param [properties] Properties to set
             * @returns ActionDiscoveryResponse instance
             */
            public static create(properties?: flowcraft.v1.IActionDiscoveryResponse): flowcraft.v1.ActionDiscoveryResponse;

            /**
             * Encodes the specified ActionDiscoveryResponse message. Does not implicitly {@link flowcraft.v1.ActionDiscoveryResponse.verify|verify} messages.
             * @param message ActionDiscoveryResponse message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: flowcraft.v1.IActionDiscoveryResponse, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified ActionDiscoveryResponse message, length delimited. Does not implicitly {@link flowcraft.v1.ActionDiscoveryResponse.verify|verify} messages.
             * @param message ActionDiscoveryResponse message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: flowcraft.v1.IActionDiscoveryResponse, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes an ActionDiscoveryResponse message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns ActionDiscoveryResponse
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): flowcraft.v1.ActionDiscoveryResponse;

            /**
             * Decodes an ActionDiscoveryResponse message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns ActionDiscoveryResponse
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): flowcraft.v1.ActionDiscoveryResponse;

            /**
             * Verifies an ActionDiscoveryResponse message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates an ActionDiscoveryResponse message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns ActionDiscoveryResponse
             */
            public static fromObject(object: { [k: string]: any }): flowcraft.v1.ActionDiscoveryResponse;

            /**
             * Creates a plain object from an ActionDiscoveryResponse message. Also converts values to other types if specified.
             * @param message ActionDiscoveryResponse
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: flowcraft.v1.ActionDiscoveryResponse, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this ActionDiscoveryResponse to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };

            /**
             * Gets the default type url for ActionDiscoveryResponse
             * @param [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns The default type url
             */
            public static getTypeUrl(typeUrlPrefix?: string): string;
        }

        /** Properties of an ActionExecutionRequest. */
        interface IActionExecutionRequest {

            /** ActionExecutionRequest actionId */
            actionId?: (string|null);

            /** ActionExecutionRequest sourceNodeId */
            sourceNodeId?: (string|null);

            /** ActionExecutionRequest contextNodeIds */
            contextNodeIds?: (string[]|null);

            /** ActionExecutionRequest paramsJson */
            paramsJson?: (string|null);
        }

        /** Represents an ActionExecutionRequest. */
        class ActionExecutionRequest implements IActionExecutionRequest {

            /**
             * Constructs a new ActionExecutionRequest.
             * @param [properties] Properties to set
             */
            constructor(properties?: flowcraft.v1.IActionExecutionRequest);

            /** ActionExecutionRequest actionId. */
            public actionId: string;

            /** ActionExecutionRequest sourceNodeId. */
            public sourceNodeId: string;

            /** ActionExecutionRequest contextNodeIds. */
            public contextNodeIds: string[];

            /** ActionExecutionRequest paramsJson. */
            public paramsJson: string;

            /**
             * Creates a new ActionExecutionRequest instance using the specified properties.
             * @param [properties] Properties to set
             * @returns ActionExecutionRequest instance
             */
            public static create(properties?: flowcraft.v1.IActionExecutionRequest): flowcraft.v1.ActionExecutionRequest;

            /**
             * Encodes the specified ActionExecutionRequest message. Does not implicitly {@link flowcraft.v1.ActionExecutionRequest.verify|verify} messages.
             * @param message ActionExecutionRequest message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: flowcraft.v1.IActionExecutionRequest, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified ActionExecutionRequest message, length delimited. Does not implicitly {@link flowcraft.v1.ActionExecutionRequest.verify|verify} messages.
             * @param message ActionExecutionRequest message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: flowcraft.v1.IActionExecutionRequest, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes an ActionExecutionRequest message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns ActionExecutionRequest
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): flowcraft.v1.ActionExecutionRequest;

            /**
             * Decodes an ActionExecutionRequest message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns ActionExecutionRequest
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): flowcraft.v1.ActionExecutionRequest;

            /**
             * Verifies an ActionExecutionRequest message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates an ActionExecutionRequest message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns ActionExecutionRequest
             */
            public static fromObject(object: { [k: string]: any }): flowcraft.v1.ActionExecutionRequest;

            /**
             * Creates a plain object from an ActionExecutionRequest message. Also converts values to other types if specified.
             * @param message ActionExecutionRequest
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: flowcraft.v1.ActionExecutionRequest, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this ActionExecutionRequest to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };

            /**
             * Gets the default type url for ActionExecutionRequest
             * @param [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns The default type url
             */
            public static getTypeUrl(typeUrlPrefix?: string): string;
        }

        /** Properties of an ActionExecutionResult. */
        interface IActionExecutionResult {

            /** ActionExecutionResult success */
            success?: (boolean|null);

            /** ActionExecutionResult taskId */
            taskId?: (string|null);

            /** ActionExecutionResult diff */
            diff?: (flowcraft.v1.IGraphDiff|null);

            /** ActionExecutionResult strategy */
            strategy?: (flowcraft.v1.ActionExecutionStrategy|null);
        }

        /** Represents an ActionExecutionResult. */
        class ActionExecutionResult implements IActionExecutionResult {

            /**
             * Constructs a new ActionExecutionResult.
             * @param [properties] Properties to set
             */
            constructor(properties?: flowcraft.v1.IActionExecutionResult);

            /** ActionExecutionResult success. */
            public success: boolean;

            /** ActionExecutionResult taskId. */
            public taskId: string;

            /** ActionExecutionResult diff. */
            public diff?: (flowcraft.v1.IGraphDiff|null);

            /** ActionExecutionResult strategy. */
            public strategy: flowcraft.v1.ActionExecutionStrategy;

            /**
             * Creates a new ActionExecutionResult instance using the specified properties.
             * @param [properties] Properties to set
             * @returns ActionExecutionResult instance
             */
            public static create(properties?: flowcraft.v1.IActionExecutionResult): flowcraft.v1.ActionExecutionResult;

            /**
             * Encodes the specified ActionExecutionResult message. Does not implicitly {@link flowcraft.v1.ActionExecutionResult.verify|verify} messages.
             * @param message ActionExecutionResult message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: flowcraft.v1.IActionExecutionResult, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified ActionExecutionResult message, length delimited. Does not implicitly {@link flowcraft.v1.ActionExecutionResult.verify|verify} messages.
             * @param message ActionExecutionResult message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: flowcraft.v1.IActionExecutionResult, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes an ActionExecutionResult message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns ActionExecutionResult
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): flowcraft.v1.ActionExecutionResult;

            /**
             * Decodes an ActionExecutionResult message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns ActionExecutionResult
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): flowcraft.v1.ActionExecutionResult;

            /**
             * Verifies an ActionExecutionResult message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates an ActionExecutionResult message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns ActionExecutionResult
             */
            public static fromObject(object: { [k: string]: any }): flowcraft.v1.ActionExecutionResult;

            /**
             * Creates a plain object from an ActionExecutionResult message. Also converts values to other types if specified.
             * @param message ActionExecutionResult
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: flowcraft.v1.ActionExecutionResult, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this ActionExecutionResult to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };

            /**
             * Gets the default type url for ActionExecutionResult
             * @param [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns The default type url
             */
            public static getTypeUrl(typeUrlPrefix?: string): string;
        }

        /** Properties of a GraphDiff. */
        interface IGraphDiff {

            /** GraphDiff nodesJson */
            nodesJson?: (string|null);

            /** GraphDiff edgesJson */
            edgesJson?: (string|null);

            /** GraphDiff removeNodeIds */
            removeNodeIds?: (string[]|null);

            /** GraphDiff removeEdgeIds */
            removeEdgeIds?: (string[]|null);
        }

        /** Represents a GraphDiff. */
        class GraphDiff implements IGraphDiff {

            /**
             * Constructs a new GraphDiff.
             * @param [properties] Properties to set
             */
            constructor(properties?: flowcraft.v1.IGraphDiff);

            /** GraphDiff nodesJson. */
            public nodesJson: string;

            /** GraphDiff edgesJson. */
            public edgesJson: string;

            /** GraphDiff removeNodeIds. */
            public removeNodeIds: string[];

            /** GraphDiff removeEdgeIds. */
            public removeEdgeIds: string[];

            /**
             * Creates a new GraphDiff instance using the specified properties.
             * @param [properties] Properties to set
             * @returns GraphDiff instance
             */
            public static create(properties?: flowcraft.v1.IGraphDiff): flowcraft.v1.GraphDiff;

            /**
             * Encodes the specified GraphDiff message. Does not implicitly {@link flowcraft.v1.GraphDiff.verify|verify} messages.
             * @param message GraphDiff message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: flowcraft.v1.IGraphDiff, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified GraphDiff message, length delimited. Does not implicitly {@link flowcraft.v1.GraphDiff.verify|verify} messages.
             * @param message GraphDiff message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: flowcraft.v1.IGraphDiff, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a GraphDiff message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns GraphDiff
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): flowcraft.v1.GraphDiff;

            /**
             * Decodes a GraphDiff message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns GraphDiff
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): flowcraft.v1.GraphDiff;

            /**
             * Verifies a GraphDiff message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates a GraphDiff message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns GraphDiff
             */
            public static fromObject(object: { [k: string]: any }): flowcraft.v1.GraphDiff;

            /**
             * Creates a plain object from a GraphDiff message. Also converts values to other types if specified.
             * @param message GraphDiff
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: flowcraft.v1.GraphDiff, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this GraphDiff to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };

            /**
             * Gets the default type url for GraphDiff
             * @param [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns The default type url
             */
            public static getTypeUrl(typeUrlPrefix?: string): string;
        }

        /** Properties of a Position. */
        interface IPosition {

            /** Position x */
            x?: (number|null);

            /** Position y */
            y?: (number|null);
        }

        /** Represents a Position. */
        class Position implements IPosition {

            /**
             * Constructs a new Position.
             * @param [properties] Properties to set
             */
            constructor(properties?: flowcraft.v1.IPosition);

            /** Position x. */
            public x: number;

            /** Position y. */
            public y: number;

            /**
             * Creates a new Position instance using the specified properties.
             * @param [properties] Properties to set
             * @returns Position instance
             */
            public static create(properties?: flowcraft.v1.IPosition): flowcraft.v1.Position;

            /**
             * Encodes the specified Position message. Does not implicitly {@link flowcraft.v1.Position.verify|verify} messages.
             * @param message Position message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: flowcraft.v1.IPosition, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified Position message, length delimited. Does not implicitly {@link flowcraft.v1.Position.verify|verify} messages.
             * @param message Position message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: flowcraft.v1.IPosition, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a Position message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns Position
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): flowcraft.v1.Position;

            /**
             * Decodes a Position message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns Position
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): flowcraft.v1.Position;

            /**
             * Verifies a Position message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates a Position message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns Position
             */
            public static fromObject(object: { [k: string]: any }): flowcraft.v1.Position;

            /**
             * Creates a plain object from a Position message. Also converts values to other types if specified.
             * @param message Position
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: flowcraft.v1.Position, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this Position to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };

            /**
             * Gets the default type url for Position
             * @param [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns The default type url
             */
            public static getTypeUrl(typeUrlPrefix?: string): string;
        }

        /** Properties of a Rect. */
        interface IRect {

            /** Rect x */
            x?: (number|null);

            /** Rect y */
            y?: (number|null);

            /** Rect width */
            width?: (number|null);

            /** Rect height */
            height?: (number|null);
        }

        /** Represents a Rect. */
        class Rect implements IRect {

            /**
             * Constructs a new Rect.
             * @param [properties] Properties to set
             */
            constructor(properties?: flowcraft.v1.IRect);

            /** Rect x. */
            public x: number;

            /** Rect y. */
            public y: number;

            /** Rect width. */
            public width: number;

            /** Rect height. */
            public height: number;

            /**
             * Creates a new Rect instance using the specified properties.
             * @param [properties] Properties to set
             * @returns Rect instance
             */
            public static create(properties?: flowcraft.v1.IRect): flowcraft.v1.Rect;

            /**
             * Encodes the specified Rect message. Does not implicitly {@link flowcraft.v1.Rect.verify|verify} messages.
             * @param message Rect message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: flowcraft.v1.IRect, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified Rect message, length delimited. Does not implicitly {@link flowcraft.v1.Rect.verify|verify} messages.
             * @param message Rect message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: flowcraft.v1.IRect, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a Rect message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns Rect
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): flowcraft.v1.Rect;

            /**
             * Decodes a Rect message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns Rect
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): flowcraft.v1.Rect;

            /**
             * Verifies a Rect message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates a Rect message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns Rect
             */
            public static fromObject(object: { [k: string]: any }): flowcraft.v1.Rect;

            /**
             * Creates a plain object from a Rect message. Also converts values to other types if specified.
             * @param message Rect
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: flowcraft.v1.Rect, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this Rect to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };

            /**
             * Gets the default type url for Rect
             * @param [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns The default type url
             */
            public static getTypeUrl(typeUrlPrefix?: string): string;
        }

        /** Properties of a Viewport. */
        interface IViewport {

            /** Viewport x */
            x?: (number|null);

            /** Viewport y */
            y?: (number|null);

            /** Viewport zoom */
            zoom?: (number|null);
        }

        /** Represents a Viewport. */
        class Viewport implements IViewport {

            /**
             * Constructs a new Viewport.
             * @param [properties] Properties to set
             */
            constructor(properties?: flowcraft.v1.IViewport);

            /** Viewport x. */
            public x: number;

            /** Viewport y. */
            public y: number;

            /** Viewport zoom. */
            public zoom: number;

            /**
             * Creates a new Viewport instance using the specified properties.
             * @param [properties] Properties to set
             * @returns Viewport instance
             */
            public static create(properties?: flowcraft.v1.IViewport): flowcraft.v1.Viewport;

            /**
             * Encodes the specified Viewport message. Does not implicitly {@link flowcraft.v1.Viewport.verify|verify} messages.
             * @param message Viewport message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: flowcraft.v1.IViewport, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified Viewport message, length delimited. Does not implicitly {@link flowcraft.v1.Viewport.verify|verify} messages.
             * @param message Viewport message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: flowcraft.v1.IViewport, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a Viewport message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns Viewport
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): flowcraft.v1.Viewport;

            /**
             * Decodes a Viewport message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns Viewport
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): flowcraft.v1.Viewport;

            /**
             * Verifies a Viewport message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates a Viewport message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns Viewport
             */
            public static fromObject(object: { [k: string]: any }): flowcraft.v1.Viewport;

            /**
             * Creates a plain object from a Viewport message. Also converts values to other types if specified.
             * @param message Viewport
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: flowcraft.v1.Viewport, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this Viewport to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };

            /**
             * Gets the default type url for Viewport
             * @param [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns The default type url
             */
            public static getTypeUrl(typeUrlPrefix?: string): string;
        }

        /** PortStyle enum. */
        enum PortStyle {
            PORT_STYLE_CIRCLE = 0,
            PORT_STYLE_SQUARE = 1,
            PORT_STYLE_DIAMOND = 2,
            PORT_STYLE_DASH = 3
        }

        /** Properties of a PortType. */
        interface IPortType {

            /** PortType mainType */
            mainType?: (string|null);

            /** PortType itemType */
            itemType?: (string|null);

            /** PortType isGeneric */
            isGeneric?: (boolean|null);
        }

        /** Represents a PortType. */
        class PortType implements IPortType {

            /**
             * Constructs a new PortType.
             * @param [properties] Properties to set
             */
            constructor(properties?: flowcraft.v1.IPortType);

            /** PortType mainType. */
            public mainType: string;

            /** PortType itemType. */
            public itemType: string;

            /** PortType isGeneric. */
            public isGeneric: boolean;

            /**
             * Creates a new PortType instance using the specified properties.
             * @param [properties] Properties to set
             * @returns PortType instance
             */
            public static create(properties?: flowcraft.v1.IPortType): flowcraft.v1.PortType;

            /**
             * Encodes the specified PortType message. Does not implicitly {@link flowcraft.v1.PortType.verify|verify} messages.
             * @param message PortType message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: flowcraft.v1.IPortType, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified PortType message, length delimited. Does not implicitly {@link flowcraft.v1.PortType.verify|verify} messages.
             * @param message PortType message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: flowcraft.v1.IPortType, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a PortType message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns PortType
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): flowcraft.v1.PortType;

            /**
             * Decodes a PortType message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns PortType
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): flowcraft.v1.PortType;

            /**
             * Verifies a PortType message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates a PortType message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns PortType
             */
            public static fromObject(object: { [k: string]: any }): flowcraft.v1.PortType;

            /**
             * Creates a plain object from a PortType message. Also converts values to other types if specified.
             * @param message PortType
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: flowcraft.v1.PortType, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this PortType to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };

            /**
             * Gets the default type url for PortType
             * @param [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns The default type url
             */
            public static getTypeUrl(typeUrlPrefix?: string): string;
        }

        /** Properties of a Port. */
        interface IPort {

            /** Port id */
            id?: (string|null);

            /** Port label */
            label?: (string|null);

            /** Port type */
            type?: (flowcraft.v1.IPortType|null);

            /** Port style */
            style?: (flowcraft.v1.PortStyle|null);

            /** Port color */
            color?: (string|null);

            /** Port description */
            description?: (string|null);
        }

        /** Represents a Port. */
        class Port implements IPort {

            /**
             * Constructs a new Port.
             * @param [properties] Properties to set
             */
            constructor(properties?: flowcraft.v1.IPort);

            /** Port id. */
            public id: string;

            /** Port label. */
            public label: string;

            /** Port type. */
            public type?: (flowcraft.v1.IPortType|null);

            /** Port style. */
            public style: flowcraft.v1.PortStyle;

            /** Port color. */
            public color: string;

            /** Port description. */
            public description: string;

            /**
             * Creates a new Port instance using the specified properties.
             * @param [properties] Properties to set
             * @returns Port instance
             */
            public static create(properties?: flowcraft.v1.IPort): flowcraft.v1.Port;

            /**
             * Encodes the specified Port message. Does not implicitly {@link flowcraft.v1.Port.verify|verify} messages.
             * @param message Port message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: flowcraft.v1.IPort, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified Port message, length delimited. Does not implicitly {@link flowcraft.v1.Port.verify|verify} messages.
             * @param message Port message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: flowcraft.v1.IPort, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a Port message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns Port
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): flowcraft.v1.Port;

            /**
             * Decodes a Port message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns Port
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): flowcraft.v1.Port;

            /**
             * Verifies a Port message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates a Port message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns Port
             */
            public static fromObject(object: { [k: string]: any }): flowcraft.v1.Port;

            /**
             * Creates a plain object from a Port message. Also converts values to other types if specified.
             * @param message Port
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: flowcraft.v1.Port, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this Port to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };

            /**
             * Gets the default type url for Port
             * @param [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns The default type url
             */
            public static getTypeUrl(typeUrlPrefix?: string): string;
        }

        /** Properties of an Edge. */
        interface IEdge {

            /** Edge id */
            id?: (string|null);

            /** Edge source */
            source?: (string|null);

            /** Edge target */
            target?: (string|null);

            /** Edge sourceHandle */
            sourceHandle?: (string|null);

            /** Edge targetHandle */
            targetHandle?: (string|null);

            /** Edge metadata */
            metadata?: ({ [k: string]: string }|null);
        }

        /** Represents an Edge. */
        class Edge implements IEdge {

            /**
             * Constructs a new Edge.
             * @param [properties] Properties to set
             */
            constructor(properties?: flowcraft.v1.IEdge);

            /** Edge id. */
            public id: string;

            /** Edge source. */
            public source: string;

            /** Edge target. */
            public target: string;

            /** Edge sourceHandle. */
            public sourceHandle: string;

            /** Edge targetHandle. */
            public targetHandle: string;

            /** Edge metadata. */
            public metadata: { [k: string]: string };

            /**
             * Creates a new Edge instance using the specified properties.
             * @param [properties] Properties to set
             * @returns Edge instance
             */
            public static create(properties?: flowcraft.v1.IEdge): flowcraft.v1.Edge;

            /**
             * Encodes the specified Edge message. Does not implicitly {@link flowcraft.v1.Edge.verify|verify} messages.
             * @param message Edge message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: flowcraft.v1.IEdge, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified Edge message, length delimited. Does not implicitly {@link flowcraft.v1.Edge.verify|verify} messages.
             * @param message Edge message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: flowcraft.v1.IEdge, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes an Edge message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns Edge
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): flowcraft.v1.Edge;

            /**
             * Decodes an Edge message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns Edge
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): flowcraft.v1.Edge;

            /**
             * Verifies an Edge message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates an Edge message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns Edge
             */
            public static fromObject(object: { [k: string]: any }): flowcraft.v1.Edge;

            /**
             * Creates a plain object from an Edge message. Also converts values to other types if specified.
             * @param message Edge
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: flowcraft.v1.Edge, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this Edge to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };

            /**
             * Gets the default type url for Edge
             * @param [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns The default type url
             */
            public static getTypeUrl(typeUrlPrefix?: string): string;
        }

        /** Properties of a FlowMessage. */
        interface IFlowMessage {

            /** FlowMessage messageId */
            messageId?: (string|null);

            /** FlowMessage timestamp */
            timestamp?: (number|Long|null);

            /** FlowMessage syncRequest */
            syncRequest?: (flowcraft.v1.ISyncRequest|null);

            /** FlowMessage yjsUpdate */
            yjsUpdate?: (Uint8Array|null);

            /** FlowMessage nodeUpdate */
            nodeUpdate?: (flowcraft.v1.IUpdateNodeRequest|null);

            /** FlowMessage widgetUpdate */
            widgetUpdate?: (flowcraft.v1.IUpdateWidgetRequest|null);

            /** FlowMessage actionExecute */
            actionExecute?: (flowcraft.v1.IActionExecutionRequest|null);

            /** FlowMessage taskCancel */
            taskCancel?: (flowcraft.v1.ITaskCancelRequest|null);

            /** FlowMessage viewportUpdate */
            viewportUpdate?: (flowcraft.v1.IViewportUpdate|null);

            /** FlowMessage snapshot */
            snapshot?: (flowcraft.v1.IGraphSnapshot|null);

            /** FlowMessage mutations */
            mutations?: (flowcraft.v1.IMutationList|null);

            /** FlowMessage taskUpdate */
            taskUpdate?: (flowcraft.v1.ITaskUpdate|null);

            /** FlowMessage streamChunk */
            streamChunk?: (flowcraft.v1.IStreamChunk|null);

            /** FlowMessage error */
            error?: (flowcraft.v1.IErrorResponse|null);
        }

        /** Represents a FlowMessage. */
        class FlowMessage implements IFlowMessage {

            /**
             * Constructs a new FlowMessage.
             * @param [properties] Properties to set
             */
            constructor(properties?: flowcraft.v1.IFlowMessage);

            /** FlowMessage messageId. */
            public messageId: string;

            /** FlowMessage timestamp. */
            public timestamp: (number|Long);

            /** FlowMessage syncRequest. */
            public syncRequest?: (flowcraft.v1.ISyncRequest|null);

            /** FlowMessage yjsUpdate. */
            public yjsUpdate?: (Uint8Array|null);

            /** FlowMessage nodeUpdate. */
            public nodeUpdate?: (flowcraft.v1.IUpdateNodeRequest|null);

            /** FlowMessage widgetUpdate. */
            public widgetUpdate?: (flowcraft.v1.IUpdateWidgetRequest|null);

            /** FlowMessage actionExecute. */
            public actionExecute?: (flowcraft.v1.IActionExecutionRequest|null);

            /** FlowMessage taskCancel. */
            public taskCancel?: (flowcraft.v1.ITaskCancelRequest|null);

            /** FlowMessage viewportUpdate. */
            public viewportUpdate?: (flowcraft.v1.IViewportUpdate|null);

            /** FlowMessage snapshot. */
            public snapshot?: (flowcraft.v1.IGraphSnapshot|null);

            /** FlowMessage mutations. */
            public mutations?: (flowcraft.v1.IMutationList|null);

            /** FlowMessage taskUpdate. */
            public taskUpdate?: (flowcraft.v1.ITaskUpdate|null);

            /** FlowMessage streamChunk. */
            public streamChunk?: (flowcraft.v1.IStreamChunk|null);

            /** FlowMessage error. */
            public error?: (flowcraft.v1.IErrorResponse|null);

            /** FlowMessage payload. */
            public payload?: ("syncRequest"|"yjsUpdate"|"nodeUpdate"|"widgetUpdate"|"actionExecute"|"taskCancel"|"viewportUpdate"|"snapshot"|"mutations"|"taskUpdate"|"streamChunk"|"error");

            /**
             * Creates a new FlowMessage instance using the specified properties.
             * @param [properties] Properties to set
             * @returns FlowMessage instance
             */
            public static create(properties?: flowcraft.v1.IFlowMessage): flowcraft.v1.FlowMessage;

            /**
             * Encodes the specified FlowMessage message. Does not implicitly {@link flowcraft.v1.FlowMessage.verify|verify} messages.
             * @param message FlowMessage message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: flowcraft.v1.IFlowMessage, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified FlowMessage message, length delimited. Does not implicitly {@link flowcraft.v1.FlowMessage.verify|verify} messages.
             * @param message FlowMessage message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: flowcraft.v1.IFlowMessage, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a FlowMessage message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns FlowMessage
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): flowcraft.v1.FlowMessage;

            /**
             * Decodes a FlowMessage message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns FlowMessage
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): flowcraft.v1.FlowMessage;

            /**
             * Verifies a FlowMessage message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates a FlowMessage message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns FlowMessage
             */
            public static fromObject(object: { [k: string]: any }): flowcraft.v1.FlowMessage;

            /**
             * Creates a plain object from a FlowMessage message. Also converts values to other types if specified.
             * @param message FlowMessage
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: flowcraft.v1.FlowMessage, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this FlowMessage to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };

            /**
             * Gets the default type url for FlowMessage
             * @param [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns The default type url
             */
            public static getTypeUrl(typeUrlPrefix?: string): string;
        }

        /** Properties of a SyncRequest. */
        interface ISyncRequest {

            /** SyncRequest graphId */
            graphId?: (string|null);
        }

        /** Represents a SyncRequest. */
        class SyncRequest implements ISyncRequest {

            /**
             * Constructs a new SyncRequest.
             * @param [properties] Properties to set
             */
            constructor(properties?: flowcraft.v1.ISyncRequest);

            /** SyncRequest graphId. */
            public graphId: string;

            /**
             * Creates a new SyncRequest instance using the specified properties.
             * @param [properties] Properties to set
             * @returns SyncRequest instance
             */
            public static create(properties?: flowcraft.v1.ISyncRequest): flowcraft.v1.SyncRequest;

            /**
             * Encodes the specified SyncRequest message. Does not implicitly {@link flowcraft.v1.SyncRequest.verify|verify} messages.
             * @param message SyncRequest message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: flowcraft.v1.ISyncRequest, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified SyncRequest message, length delimited. Does not implicitly {@link flowcraft.v1.SyncRequest.verify|verify} messages.
             * @param message SyncRequest message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: flowcraft.v1.ISyncRequest, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a SyncRequest message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns SyncRequest
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): flowcraft.v1.SyncRequest;

            /**
             * Decodes a SyncRequest message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns SyncRequest
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): flowcraft.v1.SyncRequest;

            /**
             * Verifies a SyncRequest message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates a SyncRequest message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns SyncRequest
             */
            public static fromObject(object: { [k: string]: any }): flowcraft.v1.SyncRequest;

            /**
             * Creates a plain object from a SyncRequest message. Also converts values to other types if specified.
             * @param message SyncRequest
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: flowcraft.v1.SyncRequest, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this SyncRequest to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };

            /**
             * Gets the default type url for SyncRequest
             * @param [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns The default type url
             */
            public static getTypeUrl(typeUrlPrefix?: string): string;
        }

        /** Properties of an UpdateNodeRequest. */
        interface IUpdateNodeRequest {

            /** UpdateNodeRequest nodeId */
            nodeId?: (string|null);

            /** UpdateNodeRequest data */
            data?: (flowcraft.v1.INodeData|null);

            /** UpdateNodeRequest position */
            position?: (flowcraft.v1.IPosition|null);
        }

        /** Represents an UpdateNodeRequest. */
        class UpdateNodeRequest implements IUpdateNodeRequest {

            /**
             * Constructs a new UpdateNodeRequest.
             * @param [properties] Properties to set
             */
            constructor(properties?: flowcraft.v1.IUpdateNodeRequest);

            /** UpdateNodeRequest nodeId. */
            public nodeId: string;

            /** UpdateNodeRequest data. */
            public data?: (flowcraft.v1.INodeData|null);

            /** UpdateNodeRequest position. */
            public position?: (flowcraft.v1.IPosition|null);

            /**
             * Creates a new UpdateNodeRequest instance using the specified properties.
             * @param [properties] Properties to set
             * @returns UpdateNodeRequest instance
             */
            public static create(properties?: flowcraft.v1.IUpdateNodeRequest): flowcraft.v1.UpdateNodeRequest;

            /**
             * Encodes the specified UpdateNodeRequest message. Does not implicitly {@link flowcraft.v1.UpdateNodeRequest.verify|verify} messages.
             * @param message UpdateNodeRequest message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: flowcraft.v1.IUpdateNodeRequest, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified UpdateNodeRequest message, length delimited. Does not implicitly {@link flowcraft.v1.UpdateNodeRequest.verify|verify} messages.
             * @param message UpdateNodeRequest message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: flowcraft.v1.IUpdateNodeRequest, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes an UpdateNodeRequest message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns UpdateNodeRequest
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): flowcraft.v1.UpdateNodeRequest;

            /**
             * Decodes an UpdateNodeRequest message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns UpdateNodeRequest
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): flowcraft.v1.UpdateNodeRequest;

            /**
             * Verifies an UpdateNodeRequest message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates an UpdateNodeRequest message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns UpdateNodeRequest
             */
            public static fromObject(object: { [k: string]: any }): flowcraft.v1.UpdateNodeRequest;

            /**
             * Creates a plain object from an UpdateNodeRequest message. Also converts values to other types if specified.
             * @param message UpdateNodeRequest
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: flowcraft.v1.UpdateNodeRequest, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this UpdateNodeRequest to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };

            /**
             * Gets the default type url for UpdateNodeRequest
             * @param [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns The default type url
             */
            public static getTypeUrl(typeUrlPrefix?: string): string;
        }

        /** Properties of an UpdateWidgetRequest. */
        interface IUpdateWidgetRequest {

            /** UpdateWidgetRequest nodeId */
            nodeId?: (string|null);

            /** UpdateWidgetRequest widgetId */
            widgetId?: (string|null);

            /** UpdateWidgetRequest valueJson */
            valueJson?: (string|null);
        }

        /** Represents an UpdateWidgetRequest. */
        class UpdateWidgetRequest implements IUpdateWidgetRequest {

            /**
             * Constructs a new UpdateWidgetRequest.
             * @param [properties] Properties to set
             */
            constructor(properties?: flowcraft.v1.IUpdateWidgetRequest);

            /** UpdateWidgetRequest nodeId. */
            public nodeId: string;

            /** UpdateWidgetRequest widgetId. */
            public widgetId: string;

            /** UpdateWidgetRequest valueJson. */
            public valueJson: string;

            /**
             * Creates a new UpdateWidgetRequest instance using the specified properties.
             * @param [properties] Properties to set
             * @returns UpdateWidgetRequest instance
             */
            public static create(properties?: flowcraft.v1.IUpdateWidgetRequest): flowcraft.v1.UpdateWidgetRequest;

            /**
             * Encodes the specified UpdateWidgetRequest message. Does not implicitly {@link flowcraft.v1.UpdateWidgetRequest.verify|verify} messages.
             * @param message UpdateWidgetRequest message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: flowcraft.v1.IUpdateWidgetRequest, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified UpdateWidgetRequest message, length delimited. Does not implicitly {@link flowcraft.v1.UpdateWidgetRequest.verify|verify} messages.
             * @param message UpdateWidgetRequest message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: flowcraft.v1.IUpdateWidgetRequest, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes an UpdateWidgetRequest message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns UpdateWidgetRequest
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): flowcraft.v1.UpdateWidgetRequest;

            /**
             * Decodes an UpdateWidgetRequest message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns UpdateWidgetRequest
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): flowcraft.v1.UpdateWidgetRequest;

            /**
             * Verifies an UpdateWidgetRequest message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates an UpdateWidgetRequest message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns UpdateWidgetRequest
             */
            public static fromObject(object: { [k: string]: any }): flowcraft.v1.UpdateWidgetRequest;

            /**
             * Creates a plain object from an UpdateWidgetRequest message. Also converts values to other types if specified.
             * @param message UpdateWidgetRequest
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: flowcraft.v1.UpdateWidgetRequest, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this UpdateWidgetRequest to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };

            /**
             * Gets the default type url for UpdateWidgetRequest
             * @param [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns The default type url
             */
            public static getTypeUrl(typeUrlPrefix?: string): string;
        }

        /** Properties of a ViewportUpdate. */
        interface IViewportUpdate {

            /** ViewportUpdate viewport */
            viewport?: (flowcraft.v1.IViewport|null);

            /** ViewportUpdate visibleBounds */
            visibleBounds?: (flowcraft.v1.IRect|null);
        }

        /** Represents a ViewportUpdate. */
        class ViewportUpdate implements IViewportUpdate {

            /**
             * Constructs a new ViewportUpdate.
             * @param [properties] Properties to set
             */
            constructor(properties?: flowcraft.v1.IViewportUpdate);

            /** ViewportUpdate viewport. */
            public viewport?: (flowcraft.v1.IViewport|null);

            /** ViewportUpdate visibleBounds. */
            public visibleBounds?: (flowcraft.v1.IRect|null);

            /**
             * Creates a new ViewportUpdate instance using the specified properties.
             * @param [properties] Properties to set
             * @returns ViewportUpdate instance
             */
            public static create(properties?: flowcraft.v1.IViewportUpdate): flowcraft.v1.ViewportUpdate;

            /**
             * Encodes the specified ViewportUpdate message. Does not implicitly {@link flowcraft.v1.ViewportUpdate.verify|verify} messages.
             * @param message ViewportUpdate message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: flowcraft.v1.IViewportUpdate, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified ViewportUpdate message, length delimited. Does not implicitly {@link flowcraft.v1.ViewportUpdate.verify|verify} messages.
             * @param message ViewportUpdate message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: flowcraft.v1.IViewportUpdate, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a ViewportUpdate message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns ViewportUpdate
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): flowcraft.v1.ViewportUpdate;

            /**
             * Decodes a ViewportUpdate message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns ViewportUpdate
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): flowcraft.v1.ViewportUpdate;

            /**
             * Verifies a ViewportUpdate message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates a ViewportUpdate message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns ViewportUpdate
             */
            public static fromObject(object: { [k: string]: any }): flowcraft.v1.ViewportUpdate;

            /**
             * Creates a plain object from a ViewportUpdate message. Also converts values to other types if specified.
             * @param message ViewportUpdate
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: flowcraft.v1.ViewportUpdate, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this ViewportUpdate to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };

            /**
             * Gets the default type url for ViewportUpdate
             * @param [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns The default type url
             */
            public static getTypeUrl(typeUrlPrefix?: string): string;
        }

        /** Properties of a MutationList. */
        interface IMutationList {

            /** MutationList mutations */
            mutations?: (flowcraft.v1.IGraphMutation[]|null);

            /** MutationList sequenceNumber */
            sequenceNumber?: (number|Long|null);
        }

        /** Represents a MutationList. */
        class MutationList implements IMutationList {

            /**
             * Constructs a new MutationList.
             * @param [properties] Properties to set
             */
            constructor(properties?: flowcraft.v1.IMutationList);

            /** MutationList mutations. */
            public mutations: flowcraft.v1.IGraphMutation[];

            /** MutationList sequenceNumber. */
            public sequenceNumber: (number|Long);

            /**
             * Creates a new MutationList instance using the specified properties.
             * @param [properties] Properties to set
             * @returns MutationList instance
             */
            public static create(properties?: flowcraft.v1.IMutationList): flowcraft.v1.MutationList;

            /**
             * Encodes the specified MutationList message. Does not implicitly {@link flowcraft.v1.MutationList.verify|verify} messages.
             * @param message MutationList message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: flowcraft.v1.IMutationList, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified MutationList message, length delimited. Does not implicitly {@link flowcraft.v1.MutationList.verify|verify} messages.
             * @param message MutationList message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: flowcraft.v1.IMutationList, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a MutationList message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns MutationList
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): flowcraft.v1.MutationList;

            /**
             * Decodes a MutationList message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns MutationList
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): flowcraft.v1.MutationList;

            /**
             * Verifies a MutationList message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates a MutationList message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns MutationList
             */
            public static fromObject(object: { [k: string]: any }): flowcraft.v1.MutationList;

            /**
             * Creates a plain object from a MutationList message. Also converts values to other types if specified.
             * @param message MutationList
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: flowcraft.v1.MutationList, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this MutationList to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };

            /**
             * Gets the default type url for MutationList
             * @param [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns The default type url
             */
            public static getTypeUrl(typeUrlPrefix?: string): string;
        }

        /** Properties of a StreamChunk. */
        interface IStreamChunk {

            /** StreamChunk nodeId */
            nodeId?: (string|null);

            /** StreamChunk widgetId */
            widgetId?: (string|null);

            /** StreamChunk chunkData */
            chunkData?: (string|null);

            /** StreamChunk isDone */
            isDone?: (boolean|null);
        }

        /** Represents a StreamChunk. */
        class StreamChunk implements IStreamChunk {

            /**
             * Constructs a new StreamChunk.
             * @param [properties] Properties to set
             */
            constructor(properties?: flowcraft.v1.IStreamChunk);

            /** StreamChunk nodeId. */
            public nodeId: string;

            /** StreamChunk widgetId. */
            public widgetId: string;

            /** StreamChunk chunkData. */
            public chunkData: string;

            /** StreamChunk isDone. */
            public isDone: boolean;

            /**
             * Creates a new StreamChunk instance using the specified properties.
             * @param [properties] Properties to set
             * @returns StreamChunk instance
             */
            public static create(properties?: flowcraft.v1.IStreamChunk): flowcraft.v1.StreamChunk;

            /**
             * Encodes the specified StreamChunk message. Does not implicitly {@link flowcraft.v1.StreamChunk.verify|verify} messages.
             * @param message StreamChunk message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: flowcraft.v1.IStreamChunk, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified StreamChunk message, length delimited. Does not implicitly {@link flowcraft.v1.StreamChunk.verify|verify} messages.
             * @param message StreamChunk message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: flowcraft.v1.IStreamChunk, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a StreamChunk message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns StreamChunk
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): flowcraft.v1.StreamChunk;

            /**
             * Decodes a StreamChunk message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns StreamChunk
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): flowcraft.v1.StreamChunk;

            /**
             * Verifies a StreamChunk message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates a StreamChunk message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns StreamChunk
             */
            public static fromObject(object: { [k: string]: any }): flowcraft.v1.StreamChunk;

            /**
             * Creates a plain object from a StreamChunk message. Also converts values to other types if specified.
             * @param message StreamChunk
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: flowcraft.v1.StreamChunk, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this StreamChunk to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };

            /**
             * Gets the default type url for StreamChunk
             * @param [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns The default type url
             */
            public static getTypeUrl(typeUrlPrefix?: string): string;
        }

        /** Properties of an ErrorResponse. */
        interface IErrorResponse {

            /** ErrorResponse code */
            code?: (string|null);

            /** ErrorResponse message */
            message?: (string|null);
        }

        /** Represents an ErrorResponse. */
        class ErrorResponse implements IErrorResponse {

            /**
             * Constructs a new ErrorResponse.
             * @param [properties] Properties to set
             */
            constructor(properties?: flowcraft.v1.IErrorResponse);

            /** ErrorResponse code. */
            public code: string;

            /** ErrorResponse message. */
            public message: string;

            /**
             * Creates a new ErrorResponse instance using the specified properties.
             * @param [properties] Properties to set
             * @returns ErrorResponse instance
             */
            public static create(properties?: flowcraft.v1.IErrorResponse): flowcraft.v1.ErrorResponse;

            /**
             * Encodes the specified ErrorResponse message. Does not implicitly {@link flowcraft.v1.ErrorResponse.verify|verify} messages.
             * @param message ErrorResponse message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: flowcraft.v1.IErrorResponse, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified ErrorResponse message, length delimited. Does not implicitly {@link flowcraft.v1.ErrorResponse.verify|verify} messages.
             * @param message ErrorResponse message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: flowcraft.v1.IErrorResponse, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes an ErrorResponse message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns ErrorResponse
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): flowcraft.v1.ErrorResponse;

            /**
             * Decodes an ErrorResponse message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns ErrorResponse
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): flowcraft.v1.ErrorResponse;

            /**
             * Verifies an ErrorResponse message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates an ErrorResponse message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns ErrorResponse
             */
            public static fromObject(object: { [k: string]: any }): flowcraft.v1.ErrorResponse;

            /**
             * Creates a plain object from an ErrorResponse message. Also converts values to other types if specified.
             * @param message ErrorResponse
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: flowcraft.v1.ErrorResponse, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this ErrorResponse to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };

            /**
             * Gets the default type url for ErrorResponse
             * @param [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns The default type url
             */
            public static getTypeUrl(typeUrlPrefix?: string): string;
        }

        /** Represents a FlowService */
        class FlowService extends $protobuf.rpc.Service {

            /**
             * Constructs a new FlowService service.
             * @param rpcImpl RPC implementation
             * @param [requestDelimited=false] Whether requests are length-delimited
             * @param [responseDelimited=false] Whether responses are length-delimited
             */
            constructor(rpcImpl: $protobuf.RPCImpl, requestDelimited?: boolean, responseDelimited?: boolean);

            /**
             * Creates new FlowService service using the specified rpc implementation.
             * @param rpcImpl RPC implementation
             * @param [requestDelimited=false] Whether requests are length-delimited
             * @param [responseDelimited=false] Whether responses are length-delimited
             * @returns RPC service. Useful where requests and/or responses are streamed.
             */
            public static create(rpcImpl: $protobuf.RPCImpl, requestDelimited?: boolean, responseDelimited?: boolean): FlowService;

            /**
             * Calls Connect.
             * @param request FlowMessage message or plain object
             * @param callback Node-style callback called with the error, if any, and FlowMessage
             */
            public connect(request: flowcraft.v1.IFlowMessage, callback: flowcraft.v1.FlowService.ConnectCallback): void;

            /**
             * Calls Connect.
             * @param request FlowMessage message or plain object
             * @returns Promise
             */
            public connect(request: flowcraft.v1.IFlowMessage): Promise<flowcraft.v1.FlowMessage>;
        }

        namespace FlowService {

            /**
             * Callback as used by {@link flowcraft.v1.FlowService#connect}.
             * @param error Error, if any
             * @param [response] FlowMessage
             */
            type ConnectCallback = (error: (Error|null), response?: flowcraft.v1.FlowMessage) => void;
        }

        /** Properties of a GraphSnapshot. */
        interface IGraphSnapshot {

            /** GraphSnapshot nodes */
            nodes?: (flowcraft.v1.INode[]|null);

            /** GraphSnapshot edges */
            edges?: (flowcraft.v1.IEdge[]|null);

            /** GraphSnapshot version */
            version?: (number|Long|null);
        }

        /** Represents a GraphSnapshot. */
        class GraphSnapshot implements IGraphSnapshot {

            /**
             * Constructs a new GraphSnapshot.
             * @param [properties] Properties to set
             */
            constructor(properties?: flowcraft.v1.IGraphSnapshot);

            /** GraphSnapshot nodes. */
            public nodes: flowcraft.v1.INode[];

            /** GraphSnapshot edges. */
            public edges: flowcraft.v1.IEdge[];

            /** GraphSnapshot version. */
            public version: (number|Long);

            /**
             * Creates a new GraphSnapshot instance using the specified properties.
             * @param [properties] Properties to set
             * @returns GraphSnapshot instance
             */
            public static create(properties?: flowcraft.v1.IGraphSnapshot): flowcraft.v1.GraphSnapshot;

            /**
             * Encodes the specified GraphSnapshot message. Does not implicitly {@link flowcraft.v1.GraphSnapshot.verify|verify} messages.
             * @param message GraphSnapshot message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: flowcraft.v1.IGraphSnapshot, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified GraphSnapshot message, length delimited. Does not implicitly {@link flowcraft.v1.GraphSnapshot.verify|verify} messages.
             * @param message GraphSnapshot message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: flowcraft.v1.IGraphSnapshot, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a GraphSnapshot message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns GraphSnapshot
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): flowcraft.v1.GraphSnapshot;

            /**
             * Decodes a GraphSnapshot message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns GraphSnapshot
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): flowcraft.v1.GraphSnapshot;

            /**
             * Verifies a GraphSnapshot message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates a GraphSnapshot message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns GraphSnapshot
             */
            public static fromObject(object: { [k: string]: any }): flowcraft.v1.GraphSnapshot;

            /**
             * Creates a plain object from a GraphSnapshot message. Also converts values to other types if specified.
             * @param message GraphSnapshot
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: flowcraft.v1.GraphSnapshot, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this GraphSnapshot to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };

            /**
             * Gets the default type url for GraphSnapshot
             * @param [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns The default type url
             */
            public static getTypeUrl(typeUrlPrefix?: string): string;
        }

        /** Properties of a GraphMutation. */
        interface IGraphMutation {

            /** GraphMutation addNode */
            addNode?: (flowcraft.v1.IAddNode|null);

            /** GraphMutation updateNode */
            updateNode?: (flowcraft.v1.IUpdateNode|null);

            /** GraphMutation removeNode */
            removeNode?: (flowcraft.v1.IRemoveNode|null);

            /** GraphMutation addEdge */
            addEdge?: (flowcraft.v1.IAddEdge|null);

            /** GraphMutation removeEdge */
            removeEdge?: (flowcraft.v1.IRemoveEdge|null);

            /** GraphMutation addSubgraph */
            addSubgraph?: (flowcraft.v1.IAddSubGraph|null);

            /** GraphMutation clearGraph */
            clearGraph?: (flowcraft.v1.IClearGraph|null);
        }

        /** Represents a GraphMutation. */
        class GraphMutation implements IGraphMutation {

            /**
             * Constructs a new GraphMutation.
             * @param [properties] Properties to set
             */
            constructor(properties?: flowcraft.v1.IGraphMutation);

            /** GraphMutation addNode. */
            public addNode?: (flowcraft.v1.IAddNode|null);

            /** GraphMutation updateNode. */
            public updateNode?: (flowcraft.v1.IUpdateNode|null);

            /** GraphMutation removeNode. */
            public removeNode?: (flowcraft.v1.IRemoveNode|null);

            /** GraphMutation addEdge. */
            public addEdge?: (flowcraft.v1.IAddEdge|null);

            /** GraphMutation removeEdge. */
            public removeEdge?: (flowcraft.v1.IRemoveEdge|null);

            /** GraphMutation addSubgraph. */
            public addSubgraph?: (flowcraft.v1.IAddSubGraph|null);

            /** GraphMutation clearGraph. */
            public clearGraph?: (flowcraft.v1.IClearGraph|null);

            /** GraphMutation operation. */
            public operation?: ("addNode"|"updateNode"|"removeNode"|"addEdge"|"removeEdge"|"addSubgraph"|"clearGraph");

            /**
             * Creates a new GraphMutation instance using the specified properties.
             * @param [properties] Properties to set
             * @returns GraphMutation instance
             */
            public static create(properties?: flowcraft.v1.IGraphMutation): flowcraft.v1.GraphMutation;

            /**
             * Encodes the specified GraphMutation message. Does not implicitly {@link flowcraft.v1.GraphMutation.verify|verify} messages.
             * @param message GraphMutation message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: flowcraft.v1.IGraphMutation, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified GraphMutation message, length delimited. Does not implicitly {@link flowcraft.v1.GraphMutation.verify|verify} messages.
             * @param message GraphMutation message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: flowcraft.v1.IGraphMutation, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a GraphMutation message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns GraphMutation
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): flowcraft.v1.GraphMutation;

            /**
             * Decodes a GraphMutation message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns GraphMutation
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): flowcraft.v1.GraphMutation;

            /**
             * Verifies a GraphMutation message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates a GraphMutation message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns GraphMutation
             */
            public static fromObject(object: { [k: string]: any }): flowcraft.v1.GraphMutation;

            /**
             * Creates a plain object from a GraphMutation message. Also converts values to other types if specified.
             * @param message GraphMutation
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: flowcraft.v1.GraphMutation, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this GraphMutation to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };

            /**
             * Gets the default type url for GraphMutation
             * @param [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns The default type url
             */
            public static getTypeUrl(typeUrlPrefix?: string): string;
        }

        /** Properties of an AddNode. */
        interface IAddNode {

            /** AddNode node */
            node?: (flowcraft.v1.INode|null);
        }

        /** Represents an AddNode. */
        class AddNode implements IAddNode {

            /**
             * Constructs a new AddNode.
             * @param [properties] Properties to set
             */
            constructor(properties?: flowcraft.v1.IAddNode);

            /** AddNode node. */
            public node?: (flowcraft.v1.INode|null);

            /**
             * Creates a new AddNode instance using the specified properties.
             * @param [properties] Properties to set
             * @returns AddNode instance
             */
            public static create(properties?: flowcraft.v1.IAddNode): flowcraft.v1.AddNode;

            /**
             * Encodes the specified AddNode message. Does not implicitly {@link flowcraft.v1.AddNode.verify|verify} messages.
             * @param message AddNode message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: flowcraft.v1.IAddNode, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified AddNode message, length delimited. Does not implicitly {@link flowcraft.v1.AddNode.verify|verify} messages.
             * @param message AddNode message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: flowcraft.v1.IAddNode, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes an AddNode message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns AddNode
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): flowcraft.v1.AddNode;

            /**
             * Decodes an AddNode message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns AddNode
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): flowcraft.v1.AddNode;

            /**
             * Verifies an AddNode message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates an AddNode message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns AddNode
             */
            public static fromObject(object: { [k: string]: any }): flowcraft.v1.AddNode;

            /**
             * Creates a plain object from an AddNode message. Also converts values to other types if specified.
             * @param message AddNode
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: flowcraft.v1.AddNode, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this AddNode to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };

            /**
             * Gets the default type url for AddNode
             * @param [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns The default type url
             */
            public static getTypeUrl(typeUrlPrefix?: string): string;
        }

        /** Properties of an UpdateNode. */
        interface IUpdateNode {

            /** UpdateNode id */
            id?: (string|null);

            /** UpdateNode data */
            data?: (flowcraft.v1.INodeData|null);

            /** UpdateNode position */
            position?: (flowcraft.v1.IPosition|null);
        }

        /** Represents an UpdateNode. */
        class UpdateNode implements IUpdateNode {

            /**
             * Constructs a new UpdateNode.
             * @param [properties] Properties to set
             */
            constructor(properties?: flowcraft.v1.IUpdateNode);

            /** UpdateNode id. */
            public id: string;

            /** UpdateNode data. */
            public data?: (flowcraft.v1.INodeData|null);

            /** UpdateNode position. */
            public position?: (flowcraft.v1.IPosition|null);

            /**
             * Creates a new UpdateNode instance using the specified properties.
             * @param [properties] Properties to set
             * @returns UpdateNode instance
             */
            public static create(properties?: flowcraft.v1.IUpdateNode): flowcraft.v1.UpdateNode;

            /**
             * Encodes the specified UpdateNode message. Does not implicitly {@link flowcraft.v1.UpdateNode.verify|verify} messages.
             * @param message UpdateNode message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: flowcraft.v1.IUpdateNode, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified UpdateNode message, length delimited. Does not implicitly {@link flowcraft.v1.UpdateNode.verify|verify} messages.
             * @param message UpdateNode message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: flowcraft.v1.IUpdateNode, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes an UpdateNode message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns UpdateNode
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): flowcraft.v1.UpdateNode;

            /**
             * Decodes an UpdateNode message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns UpdateNode
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): flowcraft.v1.UpdateNode;

            /**
             * Verifies an UpdateNode message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates an UpdateNode message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns UpdateNode
             */
            public static fromObject(object: { [k: string]: any }): flowcraft.v1.UpdateNode;

            /**
             * Creates a plain object from an UpdateNode message. Also converts values to other types if specified.
             * @param message UpdateNode
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: flowcraft.v1.UpdateNode, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this UpdateNode to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };

            /**
             * Gets the default type url for UpdateNode
             * @param [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns The default type url
             */
            public static getTypeUrl(typeUrlPrefix?: string): string;
        }

        /** Properties of a RemoveNode. */
        interface IRemoveNode {

            /** RemoveNode id */
            id?: (string|null);
        }

        /** Represents a RemoveNode. */
        class RemoveNode implements IRemoveNode {

            /**
             * Constructs a new RemoveNode.
             * @param [properties] Properties to set
             */
            constructor(properties?: flowcraft.v1.IRemoveNode);

            /** RemoveNode id. */
            public id: string;

            /**
             * Creates a new RemoveNode instance using the specified properties.
             * @param [properties] Properties to set
             * @returns RemoveNode instance
             */
            public static create(properties?: flowcraft.v1.IRemoveNode): flowcraft.v1.RemoveNode;

            /**
             * Encodes the specified RemoveNode message. Does not implicitly {@link flowcraft.v1.RemoveNode.verify|verify} messages.
             * @param message RemoveNode message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: flowcraft.v1.IRemoveNode, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified RemoveNode message, length delimited. Does not implicitly {@link flowcraft.v1.RemoveNode.verify|verify} messages.
             * @param message RemoveNode message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: flowcraft.v1.IRemoveNode, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a RemoveNode message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns RemoveNode
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): flowcraft.v1.RemoveNode;

            /**
             * Decodes a RemoveNode message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns RemoveNode
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): flowcraft.v1.RemoveNode;

            /**
             * Verifies a RemoveNode message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates a RemoveNode message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns RemoveNode
             */
            public static fromObject(object: { [k: string]: any }): flowcraft.v1.RemoveNode;

            /**
             * Creates a plain object from a RemoveNode message. Also converts values to other types if specified.
             * @param message RemoveNode
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: flowcraft.v1.RemoveNode, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this RemoveNode to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };

            /**
             * Gets the default type url for RemoveNode
             * @param [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns The default type url
             */
            public static getTypeUrl(typeUrlPrefix?: string): string;
        }

        /** Properties of an AddEdge. */
        interface IAddEdge {

            /** AddEdge edge */
            edge?: (flowcraft.v1.IEdge|null);
        }

        /** Represents an AddEdge. */
        class AddEdge implements IAddEdge {

            /**
             * Constructs a new AddEdge.
             * @param [properties] Properties to set
             */
            constructor(properties?: flowcraft.v1.IAddEdge);

            /** AddEdge edge. */
            public edge?: (flowcraft.v1.IEdge|null);

            /**
             * Creates a new AddEdge instance using the specified properties.
             * @param [properties] Properties to set
             * @returns AddEdge instance
             */
            public static create(properties?: flowcraft.v1.IAddEdge): flowcraft.v1.AddEdge;

            /**
             * Encodes the specified AddEdge message. Does not implicitly {@link flowcraft.v1.AddEdge.verify|verify} messages.
             * @param message AddEdge message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: flowcraft.v1.IAddEdge, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified AddEdge message, length delimited. Does not implicitly {@link flowcraft.v1.AddEdge.verify|verify} messages.
             * @param message AddEdge message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: flowcraft.v1.IAddEdge, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes an AddEdge message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns AddEdge
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): flowcraft.v1.AddEdge;

            /**
             * Decodes an AddEdge message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns AddEdge
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): flowcraft.v1.AddEdge;

            /**
             * Verifies an AddEdge message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates an AddEdge message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns AddEdge
             */
            public static fromObject(object: { [k: string]: any }): flowcraft.v1.AddEdge;

            /**
             * Creates a plain object from an AddEdge message. Also converts values to other types if specified.
             * @param message AddEdge
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: flowcraft.v1.AddEdge, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this AddEdge to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };

            /**
             * Gets the default type url for AddEdge
             * @param [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns The default type url
             */
            public static getTypeUrl(typeUrlPrefix?: string): string;
        }

        /** Properties of a RemoveEdge. */
        interface IRemoveEdge {

            /** RemoveEdge id */
            id?: (string|null);
        }

        /** Represents a RemoveEdge. */
        class RemoveEdge implements IRemoveEdge {

            /**
             * Constructs a new RemoveEdge.
             * @param [properties] Properties to set
             */
            constructor(properties?: flowcraft.v1.IRemoveEdge);

            /** RemoveEdge id. */
            public id: string;

            /**
             * Creates a new RemoveEdge instance using the specified properties.
             * @param [properties] Properties to set
             * @returns RemoveEdge instance
             */
            public static create(properties?: flowcraft.v1.IRemoveEdge): flowcraft.v1.RemoveEdge;

            /**
             * Encodes the specified RemoveEdge message. Does not implicitly {@link flowcraft.v1.RemoveEdge.verify|verify} messages.
             * @param message RemoveEdge message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: flowcraft.v1.IRemoveEdge, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified RemoveEdge message, length delimited. Does not implicitly {@link flowcraft.v1.RemoveEdge.verify|verify} messages.
             * @param message RemoveEdge message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: flowcraft.v1.IRemoveEdge, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a RemoveEdge message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns RemoveEdge
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): flowcraft.v1.RemoveEdge;

            /**
             * Decodes a RemoveEdge message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns RemoveEdge
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): flowcraft.v1.RemoveEdge;

            /**
             * Verifies a RemoveEdge message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates a RemoveEdge message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns RemoveEdge
             */
            public static fromObject(object: { [k: string]: any }): flowcraft.v1.RemoveEdge;

            /**
             * Creates a plain object from a RemoveEdge message. Also converts values to other types if specified.
             * @param message RemoveEdge
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: flowcraft.v1.RemoveEdge, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this RemoveEdge to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };

            /**
             * Gets the default type url for RemoveEdge
             * @param [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns The default type url
             */
            public static getTypeUrl(typeUrlPrefix?: string): string;
        }

        /** Properties of an AddSubGraph. */
        interface IAddSubGraph {

            /** AddSubGraph nodes */
            nodes?: (flowcraft.v1.INode[]|null);

            /** AddSubGraph edges */
            edges?: (flowcraft.v1.IEdge[]|null);
        }

        /** Represents an AddSubGraph. */
        class AddSubGraph implements IAddSubGraph {

            /**
             * Constructs a new AddSubGraph.
             * @param [properties] Properties to set
             */
            constructor(properties?: flowcraft.v1.IAddSubGraph);

            /** AddSubGraph nodes. */
            public nodes: flowcraft.v1.INode[];

            /** AddSubGraph edges. */
            public edges: flowcraft.v1.IEdge[];

            /**
             * Creates a new AddSubGraph instance using the specified properties.
             * @param [properties] Properties to set
             * @returns AddSubGraph instance
             */
            public static create(properties?: flowcraft.v1.IAddSubGraph): flowcraft.v1.AddSubGraph;

            /**
             * Encodes the specified AddSubGraph message. Does not implicitly {@link flowcraft.v1.AddSubGraph.verify|verify} messages.
             * @param message AddSubGraph message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: flowcraft.v1.IAddSubGraph, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified AddSubGraph message, length delimited. Does not implicitly {@link flowcraft.v1.AddSubGraph.verify|verify} messages.
             * @param message AddSubGraph message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: flowcraft.v1.IAddSubGraph, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes an AddSubGraph message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns AddSubGraph
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): flowcraft.v1.AddSubGraph;

            /**
             * Decodes an AddSubGraph message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns AddSubGraph
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): flowcraft.v1.AddSubGraph;

            /**
             * Verifies an AddSubGraph message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates an AddSubGraph message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns AddSubGraph
             */
            public static fromObject(object: { [k: string]: any }): flowcraft.v1.AddSubGraph;

            /**
             * Creates a plain object from an AddSubGraph message. Also converts values to other types if specified.
             * @param message AddSubGraph
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: flowcraft.v1.AddSubGraph, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this AddSubGraph to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };

            /**
             * Gets the default type url for AddSubGraph
             * @param [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns The default type url
             */
            public static getTypeUrl(typeUrlPrefix?: string): string;
        }

        /** Properties of a ClearGraph. */
        interface IClearGraph {
        }

        /** Represents a ClearGraph. */
        class ClearGraph implements IClearGraph {

            /**
             * Constructs a new ClearGraph.
             * @param [properties] Properties to set
             */
            constructor(properties?: flowcraft.v1.IClearGraph);

            /**
             * Creates a new ClearGraph instance using the specified properties.
             * @param [properties] Properties to set
             * @returns ClearGraph instance
             */
            public static create(properties?: flowcraft.v1.IClearGraph): flowcraft.v1.ClearGraph;

            /**
             * Encodes the specified ClearGraph message. Does not implicitly {@link flowcraft.v1.ClearGraph.verify|verify} messages.
             * @param message ClearGraph message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: flowcraft.v1.IClearGraph, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified ClearGraph message, length delimited. Does not implicitly {@link flowcraft.v1.ClearGraph.verify|verify} messages.
             * @param message ClearGraph message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: flowcraft.v1.IClearGraph, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a ClearGraph message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns ClearGraph
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): flowcraft.v1.ClearGraph;

            /**
             * Decodes a ClearGraph message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns ClearGraph
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): flowcraft.v1.ClearGraph;

            /**
             * Verifies a ClearGraph message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates a ClearGraph message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns ClearGraph
             */
            public static fromObject(object: { [k: string]: any }): flowcraft.v1.ClearGraph;

            /**
             * Creates a plain object from a ClearGraph message. Also converts values to other types if specified.
             * @param message ClearGraph
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: flowcraft.v1.ClearGraph, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this ClearGraph to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };

            /**
             * Gets the default type url for ClearGraph
             * @param [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns The default type url
             */
            public static getTypeUrl(typeUrlPrefix?: string): string;
        }

        /** Properties of a TaskUpdate. */
        interface ITaskUpdate {

            /** TaskUpdate taskId */
            taskId?: (string|null);

            /** TaskUpdate status */
            status?: (flowcraft.v1.TaskStatus|null);

            /** TaskUpdate progress */
            progress?: (number|null);

            /** TaskUpdate message */
            message?: (string|null);

            /** TaskUpdate resultJson */
            resultJson?: (string|null);
        }

        /** Represents a TaskUpdate. */
        class TaskUpdate implements ITaskUpdate {

            /**
             * Constructs a new TaskUpdate.
             * @param [properties] Properties to set
             */
            constructor(properties?: flowcraft.v1.ITaskUpdate);

            /** TaskUpdate taskId. */
            public taskId: string;

            /** TaskUpdate status. */
            public status: flowcraft.v1.TaskStatus;

            /** TaskUpdate progress. */
            public progress: number;

            /** TaskUpdate message. */
            public message: string;

            /** TaskUpdate resultJson. */
            public resultJson: string;

            /**
             * Creates a new TaskUpdate instance using the specified properties.
             * @param [properties] Properties to set
             * @returns TaskUpdate instance
             */
            public static create(properties?: flowcraft.v1.ITaskUpdate): flowcraft.v1.TaskUpdate;

            /**
             * Encodes the specified TaskUpdate message. Does not implicitly {@link flowcraft.v1.TaskUpdate.verify|verify} messages.
             * @param message TaskUpdate message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: flowcraft.v1.ITaskUpdate, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified TaskUpdate message, length delimited. Does not implicitly {@link flowcraft.v1.TaskUpdate.verify|verify} messages.
             * @param message TaskUpdate message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: flowcraft.v1.ITaskUpdate, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a TaskUpdate message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns TaskUpdate
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): flowcraft.v1.TaskUpdate;

            /**
             * Decodes a TaskUpdate message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns TaskUpdate
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): flowcraft.v1.TaskUpdate;

            /**
             * Verifies a TaskUpdate message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates a TaskUpdate message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns TaskUpdate
             */
            public static fromObject(object: { [k: string]: any }): flowcraft.v1.TaskUpdate;

            /**
             * Creates a plain object from a TaskUpdate message. Also converts values to other types if specified.
             * @param message TaskUpdate
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: flowcraft.v1.TaskUpdate, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this TaskUpdate to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };

            /**
             * Gets the default type url for TaskUpdate
             * @param [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns The default type url
             */
            public static getTypeUrl(typeUrlPrefix?: string): string;
        }

        /** TaskStatus enum. */
        enum TaskStatus {
            TASK_PENDING = 0,
            TASK_PROCESSING = 1,
            TASK_COMPLETED = 2,
            TASK_FAILED = 3,
            TASK_CANCELLED = 4
        }

        /** Properties of a TaskCancelRequest. */
        interface ITaskCancelRequest {

            /** TaskCancelRequest taskId */
            taskId?: (string|null);
        }

        /** Represents a TaskCancelRequest. */
        class TaskCancelRequest implements ITaskCancelRequest {

            /**
             * Constructs a new TaskCancelRequest.
             * @param [properties] Properties to set
             */
            constructor(properties?: flowcraft.v1.ITaskCancelRequest);

            /** TaskCancelRequest taskId. */
            public taskId: string;

            /**
             * Creates a new TaskCancelRequest instance using the specified properties.
             * @param [properties] Properties to set
             * @returns TaskCancelRequest instance
             */
            public static create(properties?: flowcraft.v1.ITaskCancelRequest): flowcraft.v1.TaskCancelRequest;

            /**
             * Encodes the specified TaskCancelRequest message. Does not implicitly {@link flowcraft.v1.TaskCancelRequest.verify|verify} messages.
             * @param message TaskCancelRequest message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: flowcraft.v1.ITaskCancelRequest, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified TaskCancelRequest message, length delimited. Does not implicitly {@link flowcraft.v1.TaskCancelRequest.verify|verify} messages.
             * @param message TaskCancelRequest message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: flowcraft.v1.ITaskCancelRequest, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a TaskCancelRequest message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns TaskCancelRequest
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): flowcraft.v1.TaskCancelRequest;

            /**
             * Decodes a TaskCancelRequest message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns TaskCancelRequest
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): flowcraft.v1.TaskCancelRequest;

            /**
             * Verifies a TaskCancelRequest message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates a TaskCancelRequest message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns TaskCancelRequest
             */
            public static fromObject(object: { [k: string]: any }): flowcraft.v1.TaskCancelRequest;

            /**
             * Creates a plain object from a TaskCancelRequest message. Also converts values to other types if specified.
             * @param message TaskCancelRequest
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: flowcraft.v1.TaskCancelRequest, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this TaskCancelRequest to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };

            /**
             * Gets the default type url for TaskCancelRequest
             * @param [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns The default type url
             */
            public static getTypeUrl(typeUrlPrefix?: string): string;
        }

        /** Properties of a Node. */
        interface INode {

            /** Node id */
            id?: (string|null);

            /** Node type */
            type?: (string|null);

            /** Node position */
            position?: (flowcraft.v1.IPosition|null);

            /** Node data */
            data?: (flowcraft.v1.INodeData|null);

            /** Node width */
            width?: (number|null);

            /** Node height */
            height?: (number|null);

            /** Node selected */
            selected?: (boolean|null);

            /** Node parentId */
            parentId?: (string|null);
        }

        /** Represents a Node. */
        class Node implements INode {

            /**
             * Constructs a new Node.
             * @param [properties] Properties to set
             */
            constructor(properties?: flowcraft.v1.INode);

            /** Node id. */
            public id: string;

            /** Node type. */
            public type: string;

            /** Node position. */
            public position?: (flowcraft.v1.IPosition|null);

            /** Node data. */
            public data?: (flowcraft.v1.INodeData|null);

            /** Node width. */
            public width: number;

            /** Node height. */
            public height: number;

            /** Node selected. */
            public selected: boolean;

            /** Node parentId. */
            public parentId: string;

            /**
             * Creates a new Node instance using the specified properties.
             * @param [properties] Properties to set
             * @returns Node instance
             */
            public static create(properties?: flowcraft.v1.INode): flowcraft.v1.Node;

            /**
             * Encodes the specified Node message. Does not implicitly {@link flowcraft.v1.Node.verify|verify} messages.
             * @param message Node message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: flowcraft.v1.INode, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified Node message, length delimited. Does not implicitly {@link flowcraft.v1.Node.verify|verify} messages.
             * @param message Node message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: flowcraft.v1.INode, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a Node message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns Node
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): flowcraft.v1.Node;

            /**
             * Decodes a Node message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns Node
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): flowcraft.v1.Node;

            /**
             * Verifies a Node message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates a Node message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns Node
             */
            public static fromObject(object: { [k: string]: any }): flowcraft.v1.Node;

            /**
             * Creates a plain object from a Node message. Also converts values to other types if specified.
             * @param message Node
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: flowcraft.v1.Node, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this Node to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };

            /**
             * Gets the default type url for Node
             * @param [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns The default type url
             */
            public static getTypeUrl(typeUrlPrefix?: string): string;
        }

        /** Properties of a NodeData. */
        interface INodeData {

            /** NodeData label */
            label?: (string|null);

            /** NodeData availableModes */
            availableModes?: (flowcraft.v1.RenderMode[]|null);

            /** NodeData activeMode */
            activeMode?: (flowcraft.v1.RenderMode|null);

            /** NodeData media */
            media?: (flowcraft.v1.IMediaContent|null);

            /** NodeData widgets */
            widgets?: (flowcraft.v1.IWidget[]|null);

            /** NodeData inputPorts */
            inputPorts?: (flowcraft.v1.IPort[]|null);

            /** NodeData outputPorts */
            outputPorts?: (flowcraft.v1.IPort[]|null);

            /** NodeData metadata */
            metadata?: ({ [k: string]: string }|null);
        }

        /** Represents a NodeData. */
        class NodeData implements INodeData {

            /**
             * Constructs a new NodeData.
             * @param [properties] Properties to set
             */
            constructor(properties?: flowcraft.v1.INodeData);

            /** NodeData label. */
            public label: string;

            /** NodeData availableModes. */
            public availableModes: flowcraft.v1.RenderMode[];

            /** NodeData activeMode. */
            public activeMode: flowcraft.v1.RenderMode;

            /** NodeData media. */
            public media?: (flowcraft.v1.IMediaContent|null);

            /** NodeData widgets. */
            public widgets: flowcraft.v1.IWidget[];

            /** NodeData inputPorts. */
            public inputPorts: flowcraft.v1.IPort[];

            /** NodeData outputPorts. */
            public outputPorts: flowcraft.v1.IPort[];

            /** NodeData metadata. */
            public metadata: { [k: string]: string };

            /**
             * Creates a new NodeData instance using the specified properties.
             * @param [properties] Properties to set
             * @returns NodeData instance
             */
            public static create(properties?: flowcraft.v1.INodeData): flowcraft.v1.NodeData;

            /**
             * Encodes the specified NodeData message. Does not implicitly {@link flowcraft.v1.NodeData.verify|verify} messages.
             * @param message NodeData message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: flowcraft.v1.INodeData, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified NodeData message, length delimited. Does not implicitly {@link flowcraft.v1.NodeData.verify|verify} messages.
             * @param message NodeData message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: flowcraft.v1.INodeData, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a NodeData message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns NodeData
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): flowcraft.v1.NodeData;

            /**
             * Decodes a NodeData message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns NodeData
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): flowcraft.v1.NodeData;

            /**
             * Verifies a NodeData message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates a NodeData message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns NodeData
             */
            public static fromObject(object: { [k: string]: any }): flowcraft.v1.NodeData;

            /**
             * Creates a plain object from a NodeData message. Also converts values to other types if specified.
             * @param message NodeData
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: flowcraft.v1.NodeData, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this NodeData to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };

            /**
             * Gets the default type url for NodeData
             * @param [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns The default type url
             */
            public static getTypeUrl(typeUrlPrefix?: string): string;
        }

        /** RenderMode enum. */
        enum RenderMode {
            MODE_UNSPECIFIED = 0,
            MODE_MEDIA = 1,
            MODE_WIDGETS = 2,
            MODE_MARKDOWN = 3
        }

        /** Properties of a MediaContent. */
        interface IMediaContent {

            /** MediaContent type */
            type?: (flowcraft.v1.MediaType|null);

            /** MediaContent url */
            url?: (string|null);

            /** MediaContent content */
            content?: (string|null);

            /** MediaContent aspectRatio */
            aspectRatio?: (number|null);

            /** MediaContent galleryUrls */
            galleryUrls?: (string[]|null);
        }

        /** Represents a MediaContent. */
        class MediaContent implements IMediaContent {

            /**
             * Constructs a new MediaContent.
             * @param [properties] Properties to set
             */
            constructor(properties?: flowcraft.v1.IMediaContent);

            /** MediaContent type. */
            public type: flowcraft.v1.MediaType;

            /** MediaContent url. */
            public url: string;

            /** MediaContent content. */
            public content: string;

            /** MediaContent aspectRatio. */
            public aspectRatio: number;

            /** MediaContent galleryUrls. */
            public galleryUrls: string[];

            /**
             * Creates a new MediaContent instance using the specified properties.
             * @param [properties] Properties to set
             * @returns MediaContent instance
             */
            public static create(properties?: flowcraft.v1.IMediaContent): flowcraft.v1.MediaContent;

            /**
             * Encodes the specified MediaContent message. Does not implicitly {@link flowcraft.v1.MediaContent.verify|verify} messages.
             * @param message MediaContent message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: flowcraft.v1.IMediaContent, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified MediaContent message, length delimited. Does not implicitly {@link flowcraft.v1.MediaContent.verify|verify} messages.
             * @param message MediaContent message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: flowcraft.v1.IMediaContent, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a MediaContent message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns MediaContent
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): flowcraft.v1.MediaContent;

            /**
             * Decodes a MediaContent message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns MediaContent
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): flowcraft.v1.MediaContent;

            /**
             * Verifies a MediaContent message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates a MediaContent message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns MediaContent
             */
            public static fromObject(object: { [k: string]: any }): flowcraft.v1.MediaContent;

            /**
             * Creates a plain object from a MediaContent message. Also converts values to other types if specified.
             * @param message MediaContent
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: flowcraft.v1.MediaContent, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this MediaContent to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };

            /**
             * Gets the default type url for MediaContent
             * @param [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns The default type url
             */
            public static getTypeUrl(typeUrlPrefix?: string): string;
        }

        /** MediaType enum. */
        enum MediaType {
            MEDIA_UNSPECIFIED = 0,
            MEDIA_IMAGE = 1,
            MEDIA_VIDEO = 2,
            MEDIA_AUDIO = 3,
            MEDIA_MARKDOWN = 4
        }

        /** WidgetType enum. */
        enum WidgetType {
            WIDGET_UNSPECIFIED = 0,
            WIDGET_TEXT = 1,
            WIDGET_SELECT = 2,
            WIDGET_CHECKBOX = 3,
            WIDGET_SLIDER = 4,
            WIDGET_BUTTON = 5
        }

        /** Properties of a WidgetOption. */
        interface IWidgetOption {

            /** WidgetOption label */
            label?: (string|null);

            /** WidgetOption value */
            value?: (string|null);

            /** WidgetOption description */
            description?: (string|null);
        }

        /** Represents a WidgetOption. */
        class WidgetOption implements IWidgetOption {

            /**
             * Constructs a new WidgetOption.
             * @param [properties] Properties to set
             */
            constructor(properties?: flowcraft.v1.IWidgetOption);

            /** WidgetOption label. */
            public label: string;

            /** WidgetOption value. */
            public value: string;

            /** WidgetOption description. */
            public description: string;

            /**
             * Creates a new WidgetOption instance using the specified properties.
             * @param [properties] Properties to set
             * @returns WidgetOption instance
             */
            public static create(properties?: flowcraft.v1.IWidgetOption): flowcraft.v1.WidgetOption;

            /**
             * Encodes the specified WidgetOption message. Does not implicitly {@link flowcraft.v1.WidgetOption.verify|verify} messages.
             * @param message WidgetOption message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: flowcraft.v1.IWidgetOption, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified WidgetOption message, length delimited. Does not implicitly {@link flowcraft.v1.WidgetOption.verify|verify} messages.
             * @param message WidgetOption message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: flowcraft.v1.IWidgetOption, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a WidgetOption message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns WidgetOption
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): flowcraft.v1.WidgetOption;

            /**
             * Decodes a WidgetOption message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns WidgetOption
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): flowcraft.v1.WidgetOption;

            /**
             * Verifies a WidgetOption message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates a WidgetOption message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns WidgetOption
             */
            public static fromObject(object: { [k: string]: any }): flowcraft.v1.WidgetOption;

            /**
             * Creates a plain object from a WidgetOption message. Also converts values to other types if specified.
             * @param message WidgetOption
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: flowcraft.v1.WidgetOption, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this WidgetOption to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };

            /**
             * Gets the default type url for WidgetOption
             * @param [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns The default type url
             */
            public static getTypeUrl(typeUrlPrefix?: string): string;
        }

        /** Properties of a WidgetConfig. */
        interface IWidgetConfig {

            /** WidgetConfig placeholder */
            placeholder?: (string|null);

            /** WidgetConfig min */
            min?: (number|null);

            /** WidgetConfig max */
            max?: (number|null);

            /** WidgetConfig step */
            step?: (number|null);

            /** WidgetConfig dynamicOptions */
            dynamicOptions?: (boolean|null);

            /** WidgetConfig actionTarget */
            actionTarget?: (string|null);
        }

        /** Represents a WidgetConfig. */
        class WidgetConfig implements IWidgetConfig {

            /**
             * Constructs a new WidgetConfig.
             * @param [properties] Properties to set
             */
            constructor(properties?: flowcraft.v1.IWidgetConfig);

            /** WidgetConfig placeholder. */
            public placeholder: string;

            /** WidgetConfig min. */
            public min: number;

            /** WidgetConfig max. */
            public max: number;

            /** WidgetConfig step. */
            public step: number;

            /** WidgetConfig dynamicOptions. */
            public dynamicOptions: boolean;

            /** WidgetConfig actionTarget. */
            public actionTarget: string;

            /**
             * Creates a new WidgetConfig instance using the specified properties.
             * @param [properties] Properties to set
             * @returns WidgetConfig instance
             */
            public static create(properties?: flowcraft.v1.IWidgetConfig): flowcraft.v1.WidgetConfig;

            /**
             * Encodes the specified WidgetConfig message. Does not implicitly {@link flowcraft.v1.WidgetConfig.verify|verify} messages.
             * @param message WidgetConfig message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: flowcraft.v1.IWidgetConfig, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified WidgetConfig message, length delimited. Does not implicitly {@link flowcraft.v1.WidgetConfig.verify|verify} messages.
             * @param message WidgetConfig message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: flowcraft.v1.IWidgetConfig, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a WidgetConfig message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns WidgetConfig
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): flowcraft.v1.WidgetConfig;

            /**
             * Decodes a WidgetConfig message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns WidgetConfig
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): flowcraft.v1.WidgetConfig;

            /**
             * Verifies a WidgetConfig message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates a WidgetConfig message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns WidgetConfig
             */
            public static fromObject(object: { [k: string]: any }): flowcraft.v1.WidgetConfig;

            /**
             * Creates a plain object from a WidgetConfig message. Also converts values to other types if specified.
             * @param message WidgetConfig
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: flowcraft.v1.WidgetConfig, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this WidgetConfig to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };

            /**
             * Gets the default type url for WidgetConfig
             * @param [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns The default type url
             */
            public static getTypeUrl(typeUrlPrefix?: string): string;
        }

        /** Properties of a Widget. */
        interface IWidget {

            /** Widget id */
            id?: (string|null);

            /** Widget type */
            type?: (flowcraft.v1.WidgetType|null);

            /** Widget label */
            label?: (string|null);

            /** Widget valueJson */
            valueJson?: (string|null);

            /** Widget config */
            config?: (flowcraft.v1.IWidgetConfig|null);

            /** Widget options */
            options?: (flowcraft.v1.IWidgetOption[]|null);

            /** Widget isReadonly */
            isReadonly?: (boolean|null);

            /** Widget isLoading */
            isLoading?: (boolean|null);

            /** Widget inputPortId */
            inputPortId?: (string|null);
        }

        /** Represents a Widget. */
        class Widget implements IWidget {

            /**
             * Constructs a new Widget.
             * @param [properties] Properties to set
             */
            constructor(properties?: flowcraft.v1.IWidget);

            /** Widget id. */
            public id: string;

            /** Widget type. */
            public type: flowcraft.v1.WidgetType;

            /** Widget label. */
            public label: string;

            /** Widget valueJson. */
            public valueJson: string;

            /** Widget config. */
            public config?: (flowcraft.v1.IWidgetConfig|null);

            /** Widget options. */
            public options: flowcraft.v1.IWidgetOption[];

            /** Widget isReadonly. */
            public isReadonly: boolean;

            /** Widget isLoading. */
            public isLoading: boolean;

            /** Widget inputPortId. */
            public inputPortId: string;

            /**
             * Creates a new Widget instance using the specified properties.
             * @param [properties] Properties to set
             * @returns Widget instance
             */
            public static create(properties?: flowcraft.v1.IWidget): flowcraft.v1.Widget;

            /**
             * Encodes the specified Widget message. Does not implicitly {@link flowcraft.v1.Widget.verify|verify} messages.
             * @param message Widget message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: flowcraft.v1.IWidget, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified Widget message, length delimited. Does not implicitly {@link flowcraft.v1.Widget.verify|verify} messages.
             * @param message Widget message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: flowcraft.v1.IWidget, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a Widget message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns Widget
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): flowcraft.v1.Widget;

            /**
             * Decodes a Widget message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns Widget
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): flowcraft.v1.Widget;

            /**
             * Verifies a Widget message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates a Widget message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns Widget
             */
            public static fromObject(object: { [k: string]: any }): flowcraft.v1.Widget;

            /**
             * Creates a plain object from a Widget message. Also converts values to other types if specified.
             * @param message Widget
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: flowcraft.v1.Widget, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this Widget to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };

            /**
             * Gets the default type url for Widget
             * @param [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns The default type url
             */
            public static getTypeUrl(typeUrlPrefix?: string): string;
        }
    }
}
