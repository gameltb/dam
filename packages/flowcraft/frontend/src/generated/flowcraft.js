/*eslint-disable block-scoped-var, id-length, no-control-regex, no-magic-numbers, no-prototype-builtins, no-redeclare, no-shadow, no-var, sort-vars*/
import * as $protobuf from "protobufjs/minimal";

// Common aliases
const $Reader = $protobuf.Reader, $Writer = $protobuf.Writer, $util = $protobuf.util;

// Exported root namespace
const $root = $protobuf.roots["default"] || ($protobuf.roots["default"] = {});

export const flowcraft = $root.flowcraft = (() => {

    /**
     * Namespace flowcraft.
     * @exports flowcraft
     * @namespace
     */
    const flowcraft = {};

    flowcraft.v1 = (function() {

        /**
         * Namespace v1.
         * @memberof flowcraft
         * @namespace
         */
        const v1 = {};

        /**
         * ActionExecutionStrategy enum.
         * @name flowcraft.v1.ActionExecutionStrategy
         * @enum {number}
         * @property {number} EXECUTION_IMMEDIATE=0 EXECUTION_IMMEDIATE value
         * @property {number} EXECUTION_TASK=1 EXECUTION_TASK value
         */
        v1.ActionExecutionStrategy = (function() {
            const valuesById = {}, values = Object.create(valuesById);
            values[valuesById[0] = "EXECUTION_IMMEDIATE"] = 0;
            values[valuesById[1] = "EXECUTION_TASK"] = 1;
            return values;
        })();

        v1.ActionTemplate = (function() {

            /**
             * Properties of an ActionTemplate.
             * @memberof flowcraft.v1
             * @interface IActionTemplate
             * @property {string|null} [id] ActionTemplate id
             * @property {string|null} [label] ActionTemplate label
             * @property {Array.<string>|null} [path] ActionTemplate path
             * @property {flowcraft.v1.ActionExecutionStrategy|null} [strategy] ActionTemplate strategy
             * @property {string|null} [description] ActionTemplate description
             * @property {string|null} [icon] ActionTemplate icon
             */

            /**
             * Constructs a new ActionTemplate.
             * @memberof flowcraft.v1
             * @classdesc Represents an ActionTemplate.
             * @implements IActionTemplate
             * @constructor
             * @param {flowcraft.v1.IActionTemplate=} [properties] Properties to set
             */
            function ActionTemplate(properties) {
                this.path = [];
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * ActionTemplate id.
             * @member {string} id
             * @memberof flowcraft.v1.ActionTemplate
             * @instance
             */
            ActionTemplate.prototype.id = "";

            /**
             * ActionTemplate label.
             * @member {string} label
             * @memberof flowcraft.v1.ActionTemplate
             * @instance
             */
            ActionTemplate.prototype.label = "";

            /**
             * ActionTemplate path.
             * @member {Array.<string>} path
             * @memberof flowcraft.v1.ActionTemplate
             * @instance
             */
            ActionTemplate.prototype.path = $util.emptyArray;

            /**
             * ActionTemplate strategy.
             * @member {flowcraft.v1.ActionExecutionStrategy} strategy
             * @memberof flowcraft.v1.ActionTemplate
             * @instance
             */
            ActionTemplate.prototype.strategy = 0;

            /**
             * ActionTemplate description.
             * @member {string} description
             * @memberof flowcraft.v1.ActionTemplate
             * @instance
             */
            ActionTemplate.prototype.description = "";

            /**
             * ActionTemplate icon.
             * @member {string} icon
             * @memberof flowcraft.v1.ActionTemplate
             * @instance
             */
            ActionTemplate.prototype.icon = "";

            /**
             * Creates a new ActionTemplate instance using the specified properties.
             * @function create
             * @memberof flowcraft.v1.ActionTemplate
             * @static
             * @param {flowcraft.v1.IActionTemplate=} [properties] Properties to set
             * @returns {flowcraft.v1.ActionTemplate} ActionTemplate instance
             */
            ActionTemplate.create = function create(properties) {
                return new ActionTemplate(properties);
            };

            /**
             * Encodes the specified ActionTemplate message. Does not implicitly {@link flowcraft.v1.ActionTemplate.verify|verify} messages.
             * @function encode
             * @memberof flowcraft.v1.ActionTemplate
             * @static
             * @param {flowcraft.v1.IActionTemplate} message ActionTemplate message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            ActionTemplate.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.id != null && Object.hasOwnProperty.call(message, "id"))
                    writer.uint32(/* id 1, wireType 2 =*/10).string(message.id);
                if (message.label != null && Object.hasOwnProperty.call(message, "label"))
                    writer.uint32(/* id 2, wireType 2 =*/18).string(message.label);
                if (message.path != null && message.path.length)
                    for (let i = 0; i < message.path.length; ++i)
                        writer.uint32(/* id 3, wireType 2 =*/26).string(message.path[i]);
                if (message.strategy != null && Object.hasOwnProperty.call(message, "strategy"))
                    writer.uint32(/* id 4, wireType 0 =*/32).int32(message.strategy);
                if (message.description != null && Object.hasOwnProperty.call(message, "description"))
                    writer.uint32(/* id 5, wireType 2 =*/42).string(message.description);
                if (message.icon != null && Object.hasOwnProperty.call(message, "icon"))
                    writer.uint32(/* id 6, wireType 2 =*/50).string(message.icon);
                return writer;
            };

            /**
             * Encodes the specified ActionTemplate message, length delimited. Does not implicitly {@link flowcraft.v1.ActionTemplate.verify|verify} messages.
             * @function encodeDelimited
             * @memberof flowcraft.v1.ActionTemplate
             * @static
             * @param {flowcraft.v1.IActionTemplate} message ActionTemplate message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            ActionTemplate.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes an ActionTemplate message from the specified reader or buffer.
             * @function decode
             * @memberof flowcraft.v1.ActionTemplate
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {flowcraft.v1.ActionTemplate} ActionTemplate
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            ActionTemplate.decode = function decode(reader, length, error) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.flowcraft.v1.ActionTemplate();
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    if (tag === error)
                        break;
                    switch (tag >>> 3) {
                    case 1: {
                            message.id = reader.string();
                            break;
                        }
                    case 2: {
                            message.label = reader.string();
                            break;
                        }
                    case 3: {
                            if (!(message.path && message.path.length))
                                message.path = [];
                            message.path.push(reader.string());
                            break;
                        }
                    case 4: {
                            message.strategy = reader.int32();
                            break;
                        }
                    case 5: {
                            message.description = reader.string();
                            break;
                        }
                    case 6: {
                            message.icon = reader.string();
                            break;
                        }
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };

            /**
             * Decodes an ActionTemplate message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof flowcraft.v1.ActionTemplate
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {flowcraft.v1.ActionTemplate} ActionTemplate
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            ActionTemplate.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies an ActionTemplate message.
             * @function verify
             * @memberof flowcraft.v1.ActionTemplate
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            ActionTemplate.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.id != null && message.hasOwnProperty("id"))
                    if (!$util.isString(message.id))
                        return "id: string expected";
                if (message.label != null && message.hasOwnProperty("label"))
                    if (!$util.isString(message.label))
                        return "label: string expected";
                if (message.path != null && message.hasOwnProperty("path")) {
                    if (!Array.isArray(message.path))
                        return "path: array expected";
                    for (let i = 0; i < message.path.length; ++i)
                        if (!$util.isString(message.path[i]))
                            return "path: string[] expected";
                }
                if (message.strategy != null && message.hasOwnProperty("strategy"))
                    switch (message.strategy) {
                    default:
                        return "strategy: enum value expected";
                    case 0:
                    case 1:
                        break;
                    }
                if (message.description != null && message.hasOwnProperty("description"))
                    if (!$util.isString(message.description))
                        return "description: string expected";
                if (message.icon != null && message.hasOwnProperty("icon"))
                    if (!$util.isString(message.icon))
                        return "icon: string expected";
                return null;
            };

            /**
             * Creates an ActionTemplate message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof flowcraft.v1.ActionTemplate
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {flowcraft.v1.ActionTemplate} ActionTemplate
             */
            ActionTemplate.fromObject = function fromObject(object) {
                if (object instanceof $root.flowcraft.v1.ActionTemplate)
                    return object;
                let message = new $root.flowcraft.v1.ActionTemplate();
                if (object.id != null)
                    message.id = String(object.id);
                if (object.label != null)
                    message.label = String(object.label);
                if (object.path) {
                    if (!Array.isArray(object.path))
                        throw TypeError(".flowcraft.v1.ActionTemplate.path: array expected");
                    message.path = [];
                    for (let i = 0; i < object.path.length; ++i)
                        message.path[i] = String(object.path[i]);
                }
                switch (object.strategy) {
                default:
                    if (typeof object.strategy === "number") {
                        message.strategy = object.strategy;
                        break;
                    }
                    break;
                case "EXECUTION_IMMEDIATE":
                case 0:
                    message.strategy = 0;
                    break;
                case "EXECUTION_TASK":
                case 1:
                    message.strategy = 1;
                    break;
                }
                if (object.description != null)
                    message.description = String(object.description);
                if (object.icon != null)
                    message.icon = String(object.icon);
                return message;
            };

            /**
             * Creates a plain object from an ActionTemplate message. Also converts values to other types if specified.
             * @function toObject
             * @memberof flowcraft.v1.ActionTemplate
             * @static
             * @param {flowcraft.v1.ActionTemplate} message ActionTemplate
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            ActionTemplate.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                let object = {};
                if (options.arrays || options.defaults)
                    object.path = [];
                if (options.defaults) {
                    object.id = "";
                    object.label = "";
                    object.strategy = options.enums === String ? "EXECUTION_IMMEDIATE" : 0;
                    object.description = "";
                    object.icon = "";
                }
                if (message.id != null && message.hasOwnProperty("id"))
                    object.id = message.id;
                if (message.label != null && message.hasOwnProperty("label"))
                    object.label = message.label;
                if (message.path && message.path.length) {
                    object.path = [];
                    for (let j = 0; j < message.path.length; ++j)
                        object.path[j] = message.path[j];
                }
                if (message.strategy != null && message.hasOwnProperty("strategy"))
                    object.strategy = options.enums === String ? $root.flowcraft.v1.ActionExecutionStrategy[message.strategy] === undefined ? message.strategy : $root.flowcraft.v1.ActionExecutionStrategy[message.strategy] : message.strategy;
                if (message.description != null && message.hasOwnProperty("description"))
                    object.description = message.description;
                if (message.icon != null && message.hasOwnProperty("icon"))
                    object.icon = message.icon;
                return object;
            };

            /**
             * Converts this ActionTemplate to JSON.
             * @function toJSON
             * @memberof flowcraft.v1.ActionTemplate
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            ActionTemplate.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };

            /**
             * Gets the default type url for ActionTemplate
             * @function getTypeUrl
             * @memberof flowcraft.v1.ActionTemplate
             * @static
             * @param {string} [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns {string} The default type url
             */
            ActionTemplate.getTypeUrl = function getTypeUrl(typeUrlPrefix) {
                if (typeUrlPrefix === undefined) {
                    typeUrlPrefix = "type.googleapis.com";
                }
                return typeUrlPrefix + "/flowcraft.v1.ActionTemplate";
            };

            return ActionTemplate;
        })();

        v1.ActionDiscoveryRequest = (function() {

            /**
             * Properties of an ActionDiscoveryRequest.
             * @memberof flowcraft.v1
             * @interface IActionDiscoveryRequest
             * @property {string|null} [nodeId] ActionDiscoveryRequest nodeId
             * @property {Array.<string>|null} [selectedNodeIds] ActionDiscoveryRequest selectedNodeIds
             */

            /**
             * Constructs a new ActionDiscoveryRequest.
             * @memberof flowcraft.v1
             * @classdesc Represents an ActionDiscoveryRequest.
             * @implements IActionDiscoveryRequest
             * @constructor
             * @param {flowcraft.v1.IActionDiscoveryRequest=} [properties] Properties to set
             */
            function ActionDiscoveryRequest(properties) {
                this.selectedNodeIds = [];
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * ActionDiscoveryRequest nodeId.
             * @member {string} nodeId
             * @memberof flowcraft.v1.ActionDiscoveryRequest
             * @instance
             */
            ActionDiscoveryRequest.prototype.nodeId = "";

            /**
             * ActionDiscoveryRequest selectedNodeIds.
             * @member {Array.<string>} selectedNodeIds
             * @memberof flowcraft.v1.ActionDiscoveryRequest
             * @instance
             */
            ActionDiscoveryRequest.prototype.selectedNodeIds = $util.emptyArray;

            /**
             * Creates a new ActionDiscoveryRequest instance using the specified properties.
             * @function create
             * @memberof flowcraft.v1.ActionDiscoveryRequest
             * @static
             * @param {flowcraft.v1.IActionDiscoveryRequest=} [properties] Properties to set
             * @returns {flowcraft.v1.ActionDiscoveryRequest} ActionDiscoveryRequest instance
             */
            ActionDiscoveryRequest.create = function create(properties) {
                return new ActionDiscoveryRequest(properties);
            };

            /**
             * Encodes the specified ActionDiscoveryRequest message. Does not implicitly {@link flowcraft.v1.ActionDiscoveryRequest.verify|verify} messages.
             * @function encode
             * @memberof flowcraft.v1.ActionDiscoveryRequest
             * @static
             * @param {flowcraft.v1.IActionDiscoveryRequest} message ActionDiscoveryRequest message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            ActionDiscoveryRequest.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.nodeId != null && Object.hasOwnProperty.call(message, "nodeId"))
                    writer.uint32(/* id 1, wireType 2 =*/10).string(message.nodeId);
                if (message.selectedNodeIds != null && message.selectedNodeIds.length)
                    for (let i = 0; i < message.selectedNodeIds.length; ++i)
                        writer.uint32(/* id 2, wireType 2 =*/18).string(message.selectedNodeIds[i]);
                return writer;
            };

            /**
             * Encodes the specified ActionDiscoveryRequest message, length delimited. Does not implicitly {@link flowcraft.v1.ActionDiscoveryRequest.verify|verify} messages.
             * @function encodeDelimited
             * @memberof flowcraft.v1.ActionDiscoveryRequest
             * @static
             * @param {flowcraft.v1.IActionDiscoveryRequest} message ActionDiscoveryRequest message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            ActionDiscoveryRequest.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes an ActionDiscoveryRequest message from the specified reader or buffer.
             * @function decode
             * @memberof flowcraft.v1.ActionDiscoveryRequest
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {flowcraft.v1.ActionDiscoveryRequest} ActionDiscoveryRequest
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            ActionDiscoveryRequest.decode = function decode(reader, length, error) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.flowcraft.v1.ActionDiscoveryRequest();
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    if (tag === error)
                        break;
                    switch (tag >>> 3) {
                    case 1: {
                            message.nodeId = reader.string();
                            break;
                        }
                    case 2: {
                            if (!(message.selectedNodeIds && message.selectedNodeIds.length))
                                message.selectedNodeIds = [];
                            message.selectedNodeIds.push(reader.string());
                            break;
                        }
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };

            /**
             * Decodes an ActionDiscoveryRequest message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof flowcraft.v1.ActionDiscoveryRequest
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {flowcraft.v1.ActionDiscoveryRequest} ActionDiscoveryRequest
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            ActionDiscoveryRequest.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies an ActionDiscoveryRequest message.
             * @function verify
             * @memberof flowcraft.v1.ActionDiscoveryRequest
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            ActionDiscoveryRequest.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.nodeId != null && message.hasOwnProperty("nodeId"))
                    if (!$util.isString(message.nodeId))
                        return "nodeId: string expected";
                if (message.selectedNodeIds != null && message.hasOwnProperty("selectedNodeIds")) {
                    if (!Array.isArray(message.selectedNodeIds))
                        return "selectedNodeIds: array expected";
                    for (let i = 0; i < message.selectedNodeIds.length; ++i)
                        if (!$util.isString(message.selectedNodeIds[i]))
                            return "selectedNodeIds: string[] expected";
                }
                return null;
            };

            /**
             * Creates an ActionDiscoveryRequest message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof flowcraft.v1.ActionDiscoveryRequest
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {flowcraft.v1.ActionDiscoveryRequest} ActionDiscoveryRequest
             */
            ActionDiscoveryRequest.fromObject = function fromObject(object) {
                if (object instanceof $root.flowcraft.v1.ActionDiscoveryRequest)
                    return object;
                let message = new $root.flowcraft.v1.ActionDiscoveryRequest();
                if (object.nodeId != null)
                    message.nodeId = String(object.nodeId);
                if (object.selectedNodeIds) {
                    if (!Array.isArray(object.selectedNodeIds))
                        throw TypeError(".flowcraft.v1.ActionDiscoveryRequest.selectedNodeIds: array expected");
                    message.selectedNodeIds = [];
                    for (let i = 0; i < object.selectedNodeIds.length; ++i)
                        message.selectedNodeIds[i] = String(object.selectedNodeIds[i]);
                }
                return message;
            };

            /**
             * Creates a plain object from an ActionDiscoveryRequest message. Also converts values to other types if specified.
             * @function toObject
             * @memberof flowcraft.v1.ActionDiscoveryRequest
             * @static
             * @param {flowcraft.v1.ActionDiscoveryRequest} message ActionDiscoveryRequest
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            ActionDiscoveryRequest.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                let object = {};
                if (options.arrays || options.defaults)
                    object.selectedNodeIds = [];
                if (options.defaults)
                    object.nodeId = "";
                if (message.nodeId != null && message.hasOwnProperty("nodeId"))
                    object.nodeId = message.nodeId;
                if (message.selectedNodeIds && message.selectedNodeIds.length) {
                    object.selectedNodeIds = [];
                    for (let j = 0; j < message.selectedNodeIds.length; ++j)
                        object.selectedNodeIds[j] = message.selectedNodeIds[j];
                }
                return object;
            };

            /**
             * Converts this ActionDiscoveryRequest to JSON.
             * @function toJSON
             * @memberof flowcraft.v1.ActionDiscoveryRequest
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            ActionDiscoveryRequest.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };

            /**
             * Gets the default type url for ActionDiscoveryRequest
             * @function getTypeUrl
             * @memberof flowcraft.v1.ActionDiscoveryRequest
             * @static
             * @param {string} [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns {string} The default type url
             */
            ActionDiscoveryRequest.getTypeUrl = function getTypeUrl(typeUrlPrefix) {
                if (typeUrlPrefix === undefined) {
                    typeUrlPrefix = "type.googleapis.com";
                }
                return typeUrlPrefix + "/flowcraft.v1.ActionDiscoveryRequest";
            };

            return ActionDiscoveryRequest;
        })();

        v1.ActionDiscoveryResponse = (function() {

            /**
             * Properties of an ActionDiscoveryResponse.
             * @memberof flowcraft.v1
             * @interface IActionDiscoveryResponse
             * @property {Array.<flowcraft.v1.IActionTemplate>|null} [actions] ActionDiscoveryResponse actions
             */

            /**
             * Constructs a new ActionDiscoveryResponse.
             * @memberof flowcraft.v1
             * @classdesc Represents an ActionDiscoveryResponse.
             * @implements IActionDiscoveryResponse
             * @constructor
             * @param {flowcraft.v1.IActionDiscoveryResponse=} [properties] Properties to set
             */
            function ActionDiscoveryResponse(properties) {
                this.actions = [];
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * ActionDiscoveryResponse actions.
             * @member {Array.<flowcraft.v1.IActionTemplate>} actions
             * @memberof flowcraft.v1.ActionDiscoveryResponse
             * @instance
             */
            ActionDiscoveryResponse.prototype.actions = $util.emptyArray;

            /**
             * Creates a new ActionDiscoveryResponse instance using the specified properties.
             * @function create
             * @memberof flowcraft.v1.ActionDiscoveryResponse
             * @static
             * @param {flowcraft.v1.IActionDiscoveryResponse=} [properties] Properties to set
             * @returns {flowcraft.v1.ActionDiscoveryResponse} ActionDiscoveryResponse instance
             */
            ActionDiscoveryResponse.create = function create(properties) {
                return new ActionDiscoveryResponse(properties);
            };

            /**
             * Encodes the specified ActionDiscoveryResponse message. Does not implicitly {@link flowcraft.v1.ActionDiscoveryResponse.verify|verify} messages.
             * @function encode
             * @memberof flowcraft.v1.ActionDiscoveryResponse
             * @static
             * @param {flowcraft.v1.IActionDiscoveryResponse} message ActionDiscoveryResponse message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            ActionDiscoveryResponse.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.actions != null && message.actions.length)
                    for (let i = 0; i < message.actions.length; ++i)
                        $root.flowcraft.v1.ActionTemplate.encode(message.actions[i], writer.uint32(/* id 1, wireType 2 =*/10).fork()).ldelim();
                return writer;
            };

            /**
             * Encodes the specified ActionDiscoveryResponse message, length delimited. Does not implicitly {@link flowcraft.v1.ActionDiscoveryResponse.verify|verify} messages.
             * @function encodeDelimited
             * @memberof flowcraft.v1.ActionDiscoveryResponse
             * @static
             * @param {flowcraft.v1.IActionDiscoveryResponse} message ActionDiscoveryResponse message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            ActionDiscoveryResponse.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes an ActionDiscoveryResponse message from the specified reader or buffer.
             * @function decode
             * @memberof flowcraft.v1.ActionDiscoveryResponse
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {flowcraft.v1.ActionDiscoveryResponse} ActionDiscoveryResponse
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            ActionDiscoveryResponse.decode = function decode(reader, length, error) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.flowcraft.v1.ActionDiscoveryResponse();
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    if (tag === error)
                        break;
                    switch (tag >>> 3) {
                    case 1: {
                            if (!(message.actions && message.actions.length))
                                message.actions = [];
                            message.actions.push($root.flowcraft.v1.ActionTemplate.decode(reader, reader.uint32()));
                            break;
                        }
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };

            /**
             * Decodes an ActionDiscoveryResponse message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof flowcraft.v1.ActionDiscoveryResponse
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {flowcraft.v1.ActionDiscoveryResponse} ActionDiscoveryResponse
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            ActionDiscoveryResponse.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies an ActionDiscoveryResponse message.
             * @function verify
             * @memberof flowcraft.v1.ActionDiscoveryResponse
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            ActionDiscoveryResponse.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.actions != null && message.hasOwnProperty("actions")) {
                    if (!Array.isArray(message.actions))
                        return "actions: array expected";
                    for (let i = 0; i < message.actions.length; ++i) {
                        let error = $root.flowcraft.v1.ActionTemplate.verify(message.actions[i]);
                        if (error)
                            return "actions." + error;
                    }
                }
                return null;
            };

            /**
             * Creates an ActionDiscoveryResponse message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof flowcraft.v1.ActionDiscoveryResponse
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {flowcraft.v1.ActionDiscoveryResponse} ActionDiscoveryResponse
             */
            ActionDiscoveryResponse.fromObject = function fromObject(object) {
                if (object instanceof $root.flowcraft.v1.ActionDiscoveryResponse)
                    return object;
                let message = new $root.flowcraft.v1.ActionDiscoveryResponse();
                if (object.actions) {
                    if (!Array.isArray(object.actions))
                        throw TypeError(".flowcraft.v1.ActionDiscoveryResponse.actions: array expected");
                    message.actions = [];
                    for (let i = 0; i < object.actions.length; ++i) {
                        if (typeof object.actions[i] !== "object")
                            throw TypeError(".flowcraft.v1.ActionDiscoveryResponse.actions: object expected");
                        message.actions[i] = $root.flowcraft.v1.ActionTemplate.fromObject(object.actions[i]);
                    }
                }
                return message;
            };

            /**
             * Creates a plain object from an ActionDiscoveryResponse message. Also converts values to other types if specified.
             * @function toObject
             * @memberof flowcraft.v1.ActionDiscoveryResponse
             * @static
             * @param {flowcraft.v1.ActionDiscoveryResponse} message ActionDiscoveryResponse
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            ActionDiscoveryResponse.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                let object = {};
                if (options.arrays || options.defaults)
                    object.actions = [];
                if (message.actions && message.actions.length) {
                    object.actions = [];
                    for (let j = 0; j < message.actions.length; ++j)
                        object.actions[j] = $root.flowcraft.v1.ActionTemplate.toObject(message.actions[j], options);
                }
                return object;
            };

            /**
             * Converts this ActionDiscoveryResponse to JSON.
             * @function toJSON
             * @memberof flowcraft.v1.ActionDiscoveryResponse
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            ActionDiscoveryResponse.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };

            /**
             * Gets the default type url for ActionDiscoveryResponse
             * @function getTypeUrl
             * @memberof flowcraft.v1.ActionDiscoveryResponse
             * @static
             * @param {string} [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns {string} The default type url
             */
            ActionDiscoveryResponse.getTypeUrl = function getTypeUrl(typeUrlPrefix) {
                if (typeUrlPrefix === undefined) {
                    typeUrlPrefix = "type.googleapis.com";
                }
                return typeUrlPrefix + "/flowcraft.v1.ActionDiscoveryResponse";
            };

            return ActionDiscoveryResponse;
        })();

        v1.ActionExecutionRequest = (function() {

            /**
             * Properties of an ActionExecutionRequest.
             * @memberof flowcraft.v1
             * @interface IActionExecutionRequest
             * @property {string|null} [actionId] ActionExecutionRequest actionId
             * @property {string|null} [sourceNodeId] ActionExecutionRequest sourceNodeId
             * @property {Array.<string>|null} [contextNodeIds] ActionExecutionRequest contextNodeIds
             * @property {string|null} [paramsJson] ActionExecutionRequest paramsJson
             */

            /**
             * Constructs a new ActionExecutionRequest.
             * @memberof flowcraft.v1
             * @classdesc Represents an ActionExecutionRequest.
             * @implements IActionExecutionRequest
             * @constructor
             * @param {flowcraft.v1.IActionExecutionRequest=} [properties] Properties to set
             */
            function ActionExecutionRequest(properties) {
                this.contextNodeIds = [];
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * ActionExecutionRequest actionId.
             * @member {string} actionId
             * @memberof flowcraft.v1.ActionExecutionRequest
             * @instance
             */
            ActionExecutionRequest.prototype.actionId = "";

            /**
             * ActionExecutionRequest sourceNodeId.
             * @member {string} sourceNodeId
             * @memberof flowcraft.v1.ActionExecutionRequest
             * @instance
             */
            ActionExecutionRequest.prototype.sourceNodeId = "";

            /**
             * ActionExecutionRequest contextNodeIds.
             * @member {Array.<string>} contextNodeIds
             * @memberof flowcraft.v1.ActionExecutionRequest
             * @instance
             */
            ActionExecutionRequest.prototype.contextNodeIds = $util.emptyArray;

            /**
             * ActionExecutionRequest paramsJson.
             * @member {string} paramsJson
             * @memberof flowcraft.v1.ActionExecutionRequest
             * @instance
             */
            ActionExecutionRequest.prototype.paramsJson = "";

            /**
             * Creates a new ActionExecutionRequest instance using the specified properties.
             * @function create
             * @memberof flowcraft.v1.ActionExecutionRequest
             * @static
             * @param {flowcraft.v1.IActionExecutionRequest=} [properties] Properties to set
             * @returns {flowcraft.v1.ActionExecutionRequest} ActionExecutionRequest instance
             */
            ActionExecutionRequest.create = function create(properties) {
                return new ActionExecutionRequest(properties);
            };

            /**
             * Encodes the specified ActionExecutionRequest message. Does not implicitly {@link flowcraft.v1.ActionExecutionRequest.verify|verify} messages.
             * @function encode
             * @memberof flowcraft.v1.ActionExecutionRequest
             * @static
             * @param {flowcraft.v1.IActionExecutionRequest} message ActionExecutionRequest message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            ActionExecutionRequest.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.actionId != null && Object.hasOwnProperty.call(message, "actionId"))
                    writer.uint32(/* id 1, wireType 2 =*/10).string(message.actionId);
                if (message.sourceNodeId != null && Object.hasOwnProperty.call(message, "sourceNodeId"))
                    writer.uint32(/* id 2, wireType 2 =*/18).string(message.sourceNodeId);
                if (message.contextNodeIds != null && message.contextNodeIds.length)
                    for (let i = 0; i < message.contextNodeIds.length; ++i)
                        writer.uint32(/* id 3, wireType 2 =*/26).string(message.contextNodeIds[i]);
                if (message.paramsJson != null && Object.hasOwnProperty.call(message, "paramsJson"))
                    writer.uint32(/* id 4, wireType 2 =*/34).string(message.paramsJson);
                return writer;
            };

            /**
             * Encodes the specified ActionExecutionRequest message, length delimited. Does not implicitly {@link flowcraft.v1.ActionExecutionRequest.verify|verify} messages.
             * @function encodeDelimited
             * @memberof flowcraft.v1.ActionExecutionRequest
             * @static
             * @param {flowcraft.v1.IActionExecutionRequest} message ActionExecutionRequest message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            ActionExecutionRequest.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes an ActionExecutionRequest message from the specified reader or buffer.
             * @function decode
             * @memberof flowcraft.v1.ActionExecutionRequest
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {flowcraft.v1.ActionExecutionRequest} ActionExecutionRequest
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            ActionExecutionRequest.decode = function decode(reader, length, error) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.flowcraft.v1.ActionExecutionRequest();
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    if (tag === error)
                        break;
                    switch (tag >>> 3) {
                    case 1: {
                            message.actionId = reader.string();
                            break;
                        }
                    case 2: {
                            message.sourceNodeId = reader.string();
                            break;
                        }
                    case 3: {
                            if (!(message.contextNodeIds && message.contextNodeIds.length))
                                message.contextNodeIds = [];
                            message.contextNodeIds.push(reader.string());
                            break;
                        }
                    case 4: {
                            message.paramsJson = reader.string();
                            break;
                        }
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };

            /**
             * Decodes an ActionExecutionRequest message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof flowcraft.v1.ActionExecutionRequest
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {flowcraft.v1.ActionExecutionRequest} ActionExecutionRequest
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            ActionExecutionRequest.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies an ActionExecutionRequest message.
             * @function verify
             * @memberof flowcraft.v1.ActionExecutionRequest
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            ActionExecutionRequest.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.actionId != null && message.hasOwnProperty("actionId"))
                    if (!$util.isString(message.actionId))
                        return "actionId: string expected";
                if (message.sourceNodeId != null && message.hasOwnProperty("sourceNodeId"))
                    if (!$util.isString(message.sourceNodeId))
                        return "sourceNodeId: string expected";
                if (message.contextNodeIds != null && message.hasOwnProperty("contextNodeIds")) {
                    if (!Array.isArray(message.contextNodeIds))
                        return "contextNodeIds: array expected";
                    for (let i = 0; i < message.contextNodeIds.length; ++i)
                        if (!$util.isString(message.contextNodeIds[i]))
                            return "contextNodeIds: string[] expected";
                }
                if (message.paramsJson != null && message.hasOwnProperty("paramsJson"))
                    if (!$util.isString(message.paramsJson))
                        return "paramsJson: string expected";
                return null;
            };

            /**
             * Creates an ActionExecutionRequest message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof flowcraft.v1.ActionExecutionRequest
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {flowcraft.v1.ActionExecutionRequest} ActionExecutionRequest
             */
            ActionExecutionRequest.fromObject = function fromObject(object) {
                if (object instanceof $root.flowcraft.v1.ActionExecutionRequest)
                    return object;
                let message = new $root.flowcraft.v1.ActionExecutionRequest();
                if (object.actionId != null)
                    message.actionId = String(object.actionId);
                if (object.sourceNodeId != null)
                    message.sourceNodeId = String(object.sourceNodeId);
                if (object.contextNodeIds) {
                    if (!Array.isArray(object.contextNodeIds))
                        throw TypeError(".flowcraft.v1.ActionExecutionRequest.contextNodeIds: array expected");
                    message.contextNodeIds = [];
                    for (let i = 0; i < object.contextNodeIds.length; ++i)
                        message.contextNodeIds[i] = String(object.contextNodeIds[i]);
                }
                if (object.paramsJson != null)
                    message.paramsJson = String(object.paramsJson);
                return message;
            };

            /**
             * Creates a plain object from an ActionExecutionRequest message. Also converts values to other types if specified.
             * @function toObject
             * @memberof flowcraft.v1.ActionExecutionRequest
             * @static
             * @param {flowcraft.v1.ActionExecutionRequest} message ActionExecutionRequest
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            ActionExecutionRequest.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                let object = {};
                if (options.arrays || options.defaults)
                    object.contextNodeIds = [];
                if (options.defaults) {
                    object.actionId = "";
                    object.sourceNodeId = "";
                    object.paramsJson = "";
                }
                if (message.actionId != null && message.hasOwnProperty("actionId"))
                    object.actionId = message.actionId;
                if (message.sourceNodeId != null && message.hasOwnProperty("sourceNodeId"))
                    object.sourceNodeId = message.sourceNodeId;
                if (message.contextNodeIds && message.contextNodeIds.length) {
                    object.contextNodeIds = [];
                    for (let j = 0; j < message.contextNodeIds.length; ++j)
                        object.contextNodeIds[j] = message.contextNodeIds[j];
                }
                if (message.paramsJson != null && message.hasOwnProperty("paramsJson"))
                    object.paramsJson = message.paramsJson;
                return object;
            };

            /**
             * Converts this ActionExecutionRequest to JSON.
             * @function toJSON
             * @memberof flowcraft.v1.ActionExecutionRequest
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            ActionExecutionRequest.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };

            /**
             * Gets the default type url for ActionExecutionRequest
             * @function getTypeUrl
             * @memberof flowcraft.v1.ActionExecutionRequest
             * @static
             * @param {string} [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns {string} The default type url
             */
            ActionExecutionRequest.getTypeUrl = function getTypeUrl(typeUrlPrefix) {
                if (typeUrlPrefix === undefined) {
                    typeUrlPrefix = "type.googleapis.com";
                }
                return typeUrlPrefix + "/flowcraft.v1.ActionExecutionRequest";
            };

            return ActionExecutionRequest;
        })();

        v1.ActionExecutionResult = (function() {

            /**
             * Properties of an ActionExecutionResult.
             * @memberof flowcraft.v1
             * @interface IActionExecutionResult
             * @property {boolean|null} [success] ActionExecutionResult success
             * @property {string|null} [taskId] ActionExecutionResult taskId
             * @property {flowcraft.v1.IGraphDiff|null} [diff] ActionExecutionResult diff
             * @property {flowcraft.v1.ActionExecutionStrategy|null} [strategy] ActionExecutionResult strategy
             */

            /**
             * Constructs a new ActionExecutionResult.
             * @memberof flowcraft.v1
             * @classdesc Represents an ActionExecutionResult.
             * @implements IActionExecutionResult
             * @constructor
             * @param {flowcraft.v1.IActionExecutionResult=} [properties] Properties to set
             */
            function ActionExecutionResult(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * ActionExecutionResult success.
             * @member {boolean} success
             * @memberof flowcraft.v1.ActionExecutionResult
             * @instance
             */
            ActionExecutionResult.prototype.success = false;

            /**
             * ActionExecutionResult taskId.
             * @member {string} taskId
             * @memberof flowcraft.v1.ActionExecutionResult
             * @instance
             */
            ActionExecutionResult.prototype.taskId = "";

            /**
             * ActionExecutionResult diff.
             * @member {flowcraft.v1.IGraphDiff|null|undefined} diff
             * @memberof flowcraft.v1.ActionExecutionResult
             * @instance
             */
            ActionExecutionResult.prototype.diff = null;

            /**
             * ActionExecutionResult strategy.
             * @member {flowcraft.v1.ActionExecutionStrategy} strategy
             * @memberof flowcraft.v1.ActionExecutionResult
             * @instance
             */
            ActionExecutionResult.prototype.strategy = 0;

            /**
             * Creates a new ActionExecutionResult instance using the specified properties.
             * @function create
             * @memberof flowcraft.v1.ActionExecutionResult
             * @static
             * @param {flowcraft.v1.IActionExecutionResult=} [properties] Properties to set
             * @returns {flowcraft.v1.ActionExecutionResult} ActionExecutionResult instance
             */
            ActionExecutionResult.create = function create(properties) {
                return new ActionExecutionResult(properties);
            };

            /**
             * Encodes the specified ActionExecutionResult message. Does not implicitly {@link flowcraft.v1.ActionExecutionResult.verify|verify} messages.
             * @function encode
             * @memberof flowcraft.v1.ActionExecutionResult
             * @static
             * @param {flowcraft.v1.IActionExecutionResult} message ActionExecutionResult message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            ActionExecutionResult.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.success != null && Object.hasOwnProperty.call(message, "success"))
                    writer.uint32(/* id 1, wireType 0 =*/8).bool(message.success);
                if (message.taskId != null && Object.hasOwnProperty.call(message, "taskId"))
                    writer.uint32(/* id 2, wireType 2 =*/18).string(message.taskId);
                if (message.diff != null && Object.hasOwnProperty.call(message, "diff"))
                    $root.flowcraft.v1.GraphDiff.encode(message.diff, writer.uint32(/* id 3, wireType 2 =*/26).fork()).ldelim();
                if (message.strategy != null && Object.hasOwnProperty.call(message, "strategy"))
                    writer.uint32(/* id 4, wireType 0 =*/32).int32(message.strategy);
                return writer;
            };

            /**
             * Encodes the specified ActionExecutionResult message, length delimited. Does not implicitly {@link flowcraft.v1.ActionExecutionResult.verify|verify} messages.
             * @function encodeDelimited
             * @memberof flowcraft.v1.ActionExecutionResult
             * @static
             * @param {flowcraft.v1.IActionExecutionResult} message ActionExecutionResult message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            ActionExecutionResult.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes an ActionExecutionResult message from the specified reader or buffer.
             * @function decode
             * @memberof flowcraft.v1.ActionExecutionResult
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {flowcraft.v1.ActionExecutionResult} ActionExecutionResult
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            ActionExecutionResult.decode = function decode(reader, length, error) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.flowcraft.v1.ActionExecutionResult();
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    if (tag === error)
                        break;
                    switch (tag >>> 3) {
                    case 1: {
                            message.success = reader.bool();
                            break;
                        }
                    case 2: {
                            message.taskId = reader.string();
                            break;
                        }
                    case 3: {
                            message.diff = $root.flowcraft.v1.GraphDiff.decode(reader, reader.uint32());
                            break;
                        }
                    case 4: {
                            message.strategy = reader.int32();
                            break;
                        }
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };

            /**
             * Decodes an ActionExecutionResult message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof flowcraft.v1.ActionExecutionResult
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {flowcraft.v1.ActionExecutionResult} ActionExecutionResult
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            ActionExecutionResult.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies an ActionExecutionResult message.
             * @function verify
             * @memberof flowcraft.v1.ActionExecutionResult
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            ActionExecutionResult.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.success != null && message.hasOwnProperty("success"))
                    if (typeof message.success !== "boolean")
                        return "success: boolean expected";
                if (message.taskId != null && message.hasOwnProperty("taskId"))
                    if (!$util.isString(message.taskId))
                        return "taskId: string expected";
                if (message.diff != null && message.hasOwnProperty("diff")) {
                    let error = $root.flowcraft.v1.GraphDiff.verify(message.diff);
                    if (error)
                        return "diff." + error;
                }
                if (message.strategy != null && message.hasOwnProperty("strategy"))
                    switch (message.strategy) {
                    default:
                        return "strategy: enum value expected";
                    case 0:
                    case 1:
                        break;
                    }
                return null;
            };

            /**
             * Creates an ActionExecutionResult message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof flowcraft.v1.ActionExecutionResult
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {flowcraft.v1.ActionExecutionResult} ActionExecutionResult
             */
            ActionExecutionResult.fromObject = function fromObject(object) {
                if (object instanceof $root.flowcraft.v1.ActionExecutionResult)
                    return object;
                let message = new $root.flowcraft.v1.ActionExecutionResult();
                if (object.success != null)
                    message.success = Boolean(object.success);
                if (object.taskId != null)
                    message.taskId = String(object.taskId);
                if (object.diff != null) {
                    if (typeof object.diff !== "object")
                        throw TypeError(".flowcraft.v1.ActionExecutionResult.diff: object expected");
                    message.diff = $root.flowcraft.v1.GraphDiff.fromObject(object.diff);
                }
                switch (object.strategy) {
                default:
                    if (typeof object.strategy === "number") {
                        message.strategy = object.strategy;
                        break;
                    }
                    break;
                case "EXECUTION_IMMEDIATE":
                case 0:
                    message.strategy = 0;
                    break;
                case "EXECUTION_TASK":
                case 1:
                    message.strategy = 1;
                    break;
                }
                return message;
            };

            /**
             * Creates a plain object from an ActionExecutionResult message. Also converts values to other types if specified.
             * @function toObject
             * @memberof flowcraft.v1.ActionExecutionResult
             * @static
             * @param {flowcraft.v1.ActionExecutionResult} message ActionExecutionResult
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            ActionExecutionResult.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                let object = {};
                if (options.defaults) {
                    object.success = false;
                    object.taskId = "";
                    object.diff = null;
                    object.strategy = options.enums === String ? "EXECUTION_IMMEDIATE" : 0;
                }
                if (message.success != null && message.hasOwnProperty("success"))
                    object.success = message.success;
                if (message.taskId != null && message.hasOwnProperty("taskId"))
                    object.taskId = message.taskId;
                if (message.diff != null && message.hasOwnProperty("diff"))
                    object.diff = $root.flowcraft.v1.GraphDiff.toObject(message.diff, options);
                if (message.strategy != null && message.hasOwnProperty("strategy"))
                    object.strategy = options.enums === String ? $root.flowcraft.v1.ActionExecutionStrategy[message.strategy] === undefined ? message.strategy : $root.flowcraft.v1.ActionExecutionStrategy[message.strategy] : message.strategy;
                return object;
            };

            /**
             * Converts this ActionExecutionResult to JSON.
             * @function toJSON
             * @memberof flowcraft.v1.ActionExecutionResult
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            ActionExecutionResult.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };

            /**
             * Gets the default type url for ActionExecutionResult
             * @function getTypeUrl
             * @memberof flowcraft.v1.ActionExecutionResult
             * @static
             * @param {string} [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns {string} The default type url
             */
            ActionExecutionResult.getTypeUrl = function getTypeUrl(typeUrlPrefix) {
                if (typeUrlPrefix === undefined) {
                    typeUrlPrefix = "type.googleapis.com";
                }
                return typeUrlPrefix + "/flowcraft.v1.ActionExecutionResult";
            };

            return ActionExecutionResult;
        })();

        v1.GraphDiff = (function() {

            /**
             * Properties of a GraphDiff.
             * @memberof flowcraft.v1
             * @interface IGraphDiff
             * @property {string|null} [nodesJson] GraphDiff nodesJson
             * @property {string|null} [edgesJson] GraphDiff edgesJson
             * @property {Array.<string>|null} [removeNodeIds] GraphDiff removeNodeIds
             * @property {Array.<string>|null} [removeEdgeIds] GraphDiff removeEdgeIds
             */

            /**
             * Constructs a new GraphDiff.
             * @memberof flowcraft.v1
             * @classdesc Represents a GraphDiff.
             * @implements IGraphDiff
             * @constructor
             * @param {flowcraft.v1.IGraphDiff=} [properties] Properties to set
             */
            function GraphDiff(properties) {
                this.removeNodeIds = [];
                this.removeEdgeIds = [];
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * GraphDiff nodesJson.
             * @member {string} nodesJson
             * @memberof flowcraft.v1.GraphDiff
             * @instance
             */
            GraphDiff.prototype.nodesJson = "";

            /**
             * GraphDiff edgesJson.
             * @member {string} edgesJson
             * @memberof flowcraft.v1.GraphDiff
             * @instance
             */
            GraphDiff.prototype.edgesJson = "";

            /**
             * GraphDiff removeNodeIds.
             * @member {Array.<string>} removeNodeIds
             * @memberof flowcraft.v1.GraphDiff
             * @instance
             */
            GraphDiff.prototype.removeNodeIds = $util.emptyArray;

            /**
             * GraphDiff removeEdgeIds.
             * @member {Array.<string>} removeEdgeIds
             * @memberof flowcraft.v1.GraphDiff
             * @instance
             */
            GraphDiff.prototype.removeEdgeIds = $util.emptyArray;

            /**
             * Creates a new GraphDiff instance using the specified properties.
             * @function create
             * @memberof flowcraft.v1.GraphDiff
             * @static
             * @param {flowcraft.v1.IGraphDiff=} [properties] Properties to set
             * @returns {flowcraft.v1.GraphDiff} GraphDiff instance
             */
            GraphDiff.create = function create(properties) {
                return new GraphDiff(properties);
            };

            /**
             * Encodes the specified GraphDiff message. Does not implicitly {@link flowcraft.v1.GraphDiff.verify|verify} messages.
             * @function encode
             * @memberof flowcraft.v1.GraphDiff
             * @static
             * @param {flowcraft.v1.IGraphDiff} message GraphDiff message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            GraphDiff.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.nodesJson != null && Object.hasOwnProperty.call(message, "nodesJson"))
                    writer.uint32(/* id 1, wireType 2 =*/10).string(message.nodesJson);
                if (message.edgesJson != null && Object.hasOwnProperty.call(message, "edgesJson"))
                    writer.uint32(/* id 2, wireType 2 =*/18).string(message.edgesJson);
                if (message.removeNodeIds != null && message.removeNodeIds.length)
                    for (let i = 0; i < message.removeNodeIds.length; ++i)
                        writer.uint32(/* id 3, wireType 2 =*/26).string(message.removeNodeIds[i]);
                if (message.removeEdgeIds != null && message.removeEdgeIds.length)
                    for (let i = 0; i < message.removeEdgeIds.length; ++i)
                        writer.uint32(/* id 4, wireType 2 =*/34).string(message.removeEdgeIds[i]);
                return writer;
            };

            /**
             * Encodes the specified GraphDiff message, length delimited. Does not implicitly {@link flowcraft.v1.GraphDiff.verify|verify} messages.
             * @function encodeDelimited
             * @memberof flowcraft.v1.GraphDiff
             * @static
             * @param {flowcraft.v1.IGraphDiff} message GraphDiff message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            GraphDiff.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes a GraphDiff message from the specified reader or buffer.
             * @function decode
             * @memberof flowcraft.v1.GraphDiff
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {flowcraft.v1.GraphDiff} GraphDiff
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            GraphDiff.decode = function decode(reader, length, error) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.flowcraft.v1.GraphDiff();
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    if (tag === error)
                        break;
                    switch (tag >>> 3) {
                    case 1: {
                            message.nodesJson = reader.string();
                            break;
                        }
                    case 2: {
                            message.edgesJson = reader.string();
                            break;
                        }
                    case 3: {
                            if (!(message.removeNodeIds && message.removeNodeIds.length))
                                message.removeNodeIds = [];
                            message.removeNodeIds.push(reader.string());
                            break;
                        }
                    case 4: {
                            if (!(message.removeEdgeIds && message.removeEdgeIds.length))
                                message.removeEdgeIds = [];
                            message.removeEdgeIds.push(reader.string());
                            break;
                        }
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };

            /**
             * Decodes a GraphDiff message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof flowcraft.v1.GraphDiff
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {flowcraft.v1.GraphDiff} GraphDiff
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            GraphDiff.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies a GraphDiff message.
             * @function verify
             * @memberof flowcraft.v1.GraphDiff
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            GraphDiff.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.nodesJson != null && message.hasOwnProperty("nodesJson"))
                    if (!$util.isString(message.nodesJson))
                        return "nodesJson: string expected";
                if (message.edgesJson != null && message.hasOwnProperty("edgesJson"))
                    if (!$util.isString(message.edgesJson))
                        return "edgesJson: string expected";
                if (message.removeNodeIds != null && message.hasOwnProperty("removeNodeIds")) {
                    if (!Array.isArray(message.removeNodeIds))
                        return "removeNodeIds: array expected";
                    for (let i = 0; i < message.removeNodeIds.length; ++i)
                        if (!$util.isString(message.removeNodeIds[i]))
                            return "removeNodeIds: string[] expected";
                }
                if (message.removeEdgeIds != null && message.hasOwnProperty("removeEdgeIds")) {
                    if (!Array.isArray(message.removeEdgeIds))
                        return "removeEdgeIds: array expected";
                    for (let i = 0; i < message.removeEdgeIds.length; ++i)
                        if (!$util.isString(message.removeEdgeIds[i]))
                            return "removeEdgeIds: string[] expected";
                }
                return null;
            };

            /**
             * Creates a GraphDiff message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof flowcraft.v1.GraphDiff
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {flowcraft.v1.GraphDiff} GraphDiff
             */
            GraphDiff.fromObject = function fromObject(object) {
                if (object instanceof $root.flowcraft.v1.GraphDiff)
                    return object;
                let message = new $root.flowcraft.v1.GraphDiff();
                if (object.nodesJson != null)
                    message.nodesJson = String(object.nodesJson);
                if (object.edgesJson != null)
                    message.edgesJson = String(object.edgesJson);
                if (object.removeNodeIds) {
                    if (!Array.isArray(object.removeNodeIds))
                        throw TypeError(".flowcraft.v1.GraphDiff.removeNodeIds: array expected");
                    message.removeNodeIds = [];
                    for (let i = 0; i < object.removeNodeIds.length; ++i)
                        message.removeNodeIds[i] = String(object.removeNodeIds[i]);
                }
                if (object.removeEdgeIds) {
                    if (!Array.isArray(object.removeEdgeIds))
                        throw TypeError(".flowcraft.v1.GraphDiff.removeEdgeIds: array expected");
                    message.removeEdgeIds = [];
                    for (let i = 0; i < object.removeEdgeIds.length; ++i)
                        message.removeEdgeIds[i] = String(object.removeEdgeIds[i]);
                }
                return message;
            };

            /**
             * Creates a plain object from a GraphDiff message. Also converts values to other types if specified.
             * @function toObject
             * @memberof flowcraft.v1.GraphDiff
             * @static
             * @param {flowcraft.v1.GraphDiff} message GraphDiff
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            GraphDiff.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                let object = {};
                if (options.arrays || options.defaults) {
                    object.removeNodeIds = [];
                    object.removeEdgeIds = [];
                }
                if (options.defaults) {
                    object.nodesJson = "";
                    object.edgesJson = "";
                }
                if (message.nodesJson != null && message.hasOwnProperty("nodesJson"))
                    object.nodesJson = message.nodesJson;
                if (message.edgesJson != null && message.hasOwnProperty("edgesJson"))
                    object.edgesJson = message.edgesJson;
                if (message.removeNodeIds && message.removeNodeIds.length) {
                    object.removeNodeIds = [];
                    for (let j = 0; j < message.removeNodeIds.length; ++j)
                        object.removeNodeIds[j] = message.removeNodeIds[j];
                }
                if (message.removeEdgeIds && message.removeEdgeIds.length) {
                    object.removeEdgeIds = [];
                    for (let j = 0; j < message.removeEdgeIds.length; ++j)
                        object.removeEdgeIds[j] = message.removeEdgeIds[j];
                }
                return object;
            };

            /**
             * Converts this GraphDiff to JSON.
             * @function toJSON
             * @memberof flowcraft.v1.GraphDiff
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            GraphDiff.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };

            /**
             * Gets the default type url for GraphDiff
             * @function getTypeUrl
             * @memberof flowcraft.v1.GraphDiff
             * @static
             * @param {string} [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns {string} The default type url
             */
            GraphDiff.getTypeUrl = function getTypeUrl(typeUrlPrefix) {
                if (typeUrlPrefix === undefined) {
                    typeUrlPrefix = "type.googleapis.com";
                }
                return typeUrlPrefix + "/flowcraft.v1.GraphDiff";
            };

            return GraphDiff;
        })();

        v1.Position = (function() {

            /**
             * Properties of a Position.
             * @memberof flowcraft.v1
             * @interface IPosition
             * @property {number|null} [x] Position x
             * @property {number|null} [y] Position y
             */

            /**
             * Constructs a new Position.
             * @memberof flowcraft.v1
             * @classdesc Represents a Position.
             * @implements IPosition
             * @constructor
             * @param {flowcraft.v1.IPosition=} [properties] Properties to set
             */
            function Position(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * Position x.
             * @member {number} x
             * @memberof flowcraft.v1.Position
             * @instance
             */
            Position.prototype.x = 0;

            /**
             * Position y.
             * @member {number} y
             * @memberof flowcraft.v1.Position
             * @instance
             */
            Position.prototype.y = 0;

            /**
             * Creates a new Position instance using the specified properties.
             * @function create
             * @memberof flowcraft.v1.Position
             * @static
             * @param {flowcraft.v1.IPosition=} [properties] Properties to set
             * @returns {flowcraft.v1.Position} Position instance
             */
            Position.create = function create(properties) {
                return new Position(properties);
            };

            /**
             * Encodes the specified Position message. Does not implicitly {@link flowcraft.v1.Position.verify|verify} messages.
             * @function encode
             * @memberof flowcraft.v1.Position
             * @static
             * @param {flowcraft.v1.IPosition} message Position message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            Position.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.x != null && Object.hasOwnProperty.call(message, "x"))
                    writer.uint32(/* id 1, wireType 1 =*/9).double(message.x);
                if (message.y != null && Object.hasOwnProperty.call(message, "y"))
                    writer.uint32(/* id 2, wireType 1 =*/17).double(message.y);
                return writer;
            };

            /**
             * Encodes the specified Position message, length delimited. Does not implicitly {@link flowcraft.v1.Position.verify|verify} messages.
             * @function encodeDelimited
             * @memberof flowcraft.v1.Position
             * @static
             * @param {flowcraft.v1.IPosition} message Position message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            Position.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes a Position message from the specified reader or buffer.
             * @function decode
             * @memberof flowcraft.v1.Position
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {flowcraft.v1.Position} Position
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            Position.decode = function decode(reader, length, error) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.flowcraft.v1.Position();
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    if (tag === error)
                        break;
                    switch (tag >>> 3) {
                    case 1: {
                            message.x = reader.double();
                            break;
                        }
                    case 2: {
                            message.y = reader.double();
                            break;
                        }
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };

            /**
             * Decodes a Position message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof flowcraft.v1.Position
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {flowcraft.v1.Position} Position
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            Position.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies a Position message.
             * @function verify
             * @memberof flowcraft.v1.Position
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            Position.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.x != null && message.hasOwnProperty("x"))
                    if (typeof message.x !== "number")
                        return "x: number expected";
                if (message.y != null && message.hasOwnProperty("y"))
                    if (typeof message.y !== "number")
                        return "y: number expected";
                return null;
            };

            /**
             * Creates a Position message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof flowcraft.v1.Position
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {flowcraft.v1.Position} Position
             */
            Position.fromObject = function fromObject(object) {
                if (object instanceof $root.flowcraft.v1.Position)
                    return object;
                let message = new $root.flowcraft.v1.Position();
                if (object.x != null)
                    message.x = Number(object.x);
                if (object.y != null)
                    message.y = Number(object.y);
                return message;
            };

            /**
             * Creates a plain object from a Position message. Also converts values to other types if specified.
             * @function toObject
             * @memberof flowcraft.v1.Position
             * @static
             * @param {flowcraft.v1.Position} message Position
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            Position.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                let object = {};
                if (options.defaults) {
                    object.x = 0;
                    object.y = 0;
                }
                if (message.x != null && message.hasOwnProperty("x"))
                    object.x = options.json && !isFinite(message.x) ? String(message.x) : message.x;
                if (message.y != null && message.hasOwnProperty("y"))
                    object.y = options.json && !isFinite(message.y) ? String(message.y) : message.y;
                return object;
            };

            /**
             * Converts this Position to JSON.
             * @function toJSON
             * @memberof flowcraft.v1.Position
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            Position.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };

            /**
             * Gets the default type url for Position
             * @function getTypeUrl
             * @memberof flowcraft.v1.Position
             * @static
             * @param {string} [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns {string} The default type url
             */
            Position.getTypeUrl = function getTypeUrl(typeUrlPrefix) {
                if (typeUrlPrefix === undefined) {
                    typeUrlPrefix = "type.googleapis.com";
                }
                return typeUrlPrefix + "/flowcraft.v1.Position";
            };

            return Position;
        })();

        v1.Rect = (function() {

            /**
             * Properties of a Rect.
             * @memberof flowcraft.v1
             * @interface IRect
             * @property {number|null} [x] Rect x
             * @property {number|null} [y] Rect y
             * @property {number|null} [width] Rect width
             * @property {number|null} [height] Rect height
             */

            /**
             * Constructs a new Rect.
             * @memberof flowcraft.v1
             * @classdesc Represents a Rect.
             * @implements IRect
             * @constructor
             * @param {flowcraft.v1.IRect=} [properties] Properties to set
             */
            function Rect(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * Rect x.
             * @member {number} x
             * @memberof flowcraft.v1.Rect
             * @instance
             */
            Rect.prototype.x = 0;

            /**
             * Rect y.
             * @member {number} y
             * @memberof flowcraft.v1.Rect
             * @instance
             */
            Rect.prototype.y = 0;

            /**
             * Rect width.
             * @member {number} width
             * @memberof flowcraft.v1.Rect
             * @instance
             */
            Rect.prototype.width = 0;

            /**
             * Rect height.
             * @member {number} height
             * @memberof flowcraft.v1.Rect
             * @instance
             */
            Rect.prototype.height = 0;

            /**
             * Creates a new Rect instance using the specified properties.
             * @function create
             * @memberof flowcraft.v1.Rect
             * @static
             * @param {flowcraft.v1.IRect=} [properties] Properties to set
             * @returns {flowcraft.v1.Rect} Rect instance
             */
            Rect.create = function create(properties) {
                return new Rect(properties);
            };

            /**
             * Encodes the specified Rect message. Does not implicitly {@link flowcraft.v1.Rect.verify|verify} messages.
             * @function encode
             * @memberof flowcraft.v1.Rect
             * @static
             * @param {flowcraft.v1.IRect} message Rect message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            Rect.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.x != null && Object.hasOwnProperty.call(message, "x"))
                    writer.uint32(/* id 1, wireType 1 =*/9).double(message.x);
                if (message.y != null && Object.hasOwnProperty.call(message, "y"))
                    writer.uint32(/* id 2, wireType 1 =*/17).double(message.y);
                if (message.width != null && Object.hasOwnProperty.call(message, "width"))
                    writer.uint32(/* id 3, wireType 1 =*/25).double(message.width);
                if (message.height != null && Object.hasOwnProperty.call(message, "height"))
                    writer.uint32(/* id 4, wireType 1 =*/33).double(message.height);
                return writer;
            };

            /**
             * Encodes the specified Rect message, length delimited. Does not implicitly {@link flowcraft.v1.Rect.verify|verify} messages.
             * @function encodeDelimited
             * @memberof flowcraft.v1.Rect
             * @static
             * @param {flowcraft.v1.IRect} message Rect message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            Rect.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes a Rect message from the specified reader or buffer.
             * @function decode
             * @memberof flowcraft.v1.Rect
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {flowcraft.v1.Rect} Rect
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            Rect.decode = function decode(reader, length, error) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.flowcraft.v1.Rect();
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    if (tag === error)
                        break;
                    switch (tag >>> 3) {
                    case 1: {
                            message.x = reader.double();
                            break;
                        }
                    case 2: {
                            message.y = reader.double();
                            break;
                        }
                    case 3: {
                            message.width = reader.double();
                            break;
                        }
                    case 4: {
                            message.height = reader.double();
                            break;
                        }
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };

            /**
             * Decodes a Rect message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof flowcraft.v1.Rect
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {flowcraft.v1.Rect} Rect
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            Rect.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies a Rect message.
             * @function verify
             * @memberof flowcraft.v1.Rect
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            Rect.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.x != null && message.hasOwnProperty("x"))
                    if (typeof message.x !== "number")
                        return "x: number expected";
                if (message.y != null && message.hasOwnProperty("y"))
                    if (typeof message.y !== "number")
                        return "y: number expected";
                if (message.width != null && message.hasOwnProperty("width"))
                    if (typeof message.width !== "number")
                        return "width: number expected";
                if (message.height != null && message.hasOwnProperty("height"))
                    if (typeof message.height !== "number")
                        return "height: number expected";
                return null;
            };

            /**
             * Creates a Rect message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof flowcraft.v1.Rect
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {flowcraft.v1.Rect} Rect
             */
            Rect.fromObject = function fromObject(object) {
                if (object instanceof $root.flowcraft.v1.Rect)
                    return object;
                let message = new $root.flowcraft.v1.Rect();
                if (object.x != null)
                    message.x = Number(object.x);
                if (object.y != null)
                    message.y = Number(object.y);
                if (object.width != null)
                    message.width = Number(object.width);
                if (object.height != null)
                    message.height = Number(object.height);
                return message;
            };

            /**
             * Creates a plain object from a Rect message. Also converts values to other types if specified.
             * @function toObject
             * @memberof flowcraft.v1.Rect
             * @static
             * @param {flowcraft.v1.Rect} message Rect
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            Rect.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                let object = {};
                if (options.defaults) {
                    object.x = 0;
                    object.y = 0;
                    object.width = 0;
                    object.height = 0;
                }
                if (message.x != null && message.hasOwnProperty("x"))
                    object.x = options.json && !isFinite(message.x) ? String(message.x) : message.x;
                if (message.y != null && message.hasOwnProperty("y"))
                    object.y = options.json && !isFinite(message.y) ? String(message.y) : message.y;
                if (message.width != null && message.hasOwnProperty("width"))
                    object.width = options.json && !isFinite(message.width) ? String(message.width) : message.width;
                if (message.height != null && message.hasOwnProperty("height"))
                    object.height = options.json && !isFinite(message.height) ? String(message.height) : message.height;
                return object;
            };

            /**
             * Converts this Rect to JSON.
             * @function toJSON
             * @memberof flowcraft.v1.Rect
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            Rect.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };

            /**
             * Gets the default type url for Rect
             * @function getTypeUrl
             * @memberof flowcraft.v1.Rect
             * @static
             * @param {string} [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns {string} The default type url
             */
            Rect.getTypeUrl = function getTypeUrl(typeUrlPrefix) {
                if (typeUrlPrefix === undefined) {
                    typeUrlPrefix = "type.googleapis.com";
                }
                return typeUrlPrefix + "/flowcraft.v1.Rect";
            };

            return Rect;
        })();

        v1.Viewport = (function() {

            /**
             * Properties of a Viewport.
             * @memberof flowcraft.v1
             * @interface IViewport
             * @property {number|null} [x] Viewport x
             * @property {number|null} [y] Viewport y
             * @property {number|null} [zoom] Viewport zoom
             */

            /**
             * Constructs a new Viewport.
             * @memberof flowcraft.v1
             * @classdesc Represents a Viewport.
             * @implements IViewport
             * @constructor
             * @param {flowcraft.v1.IViewport=} [properties] Properties to set
             */
            function Viewport(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * Viewport x.
             * @member {number} x
             * @memberof flowcraft.v1.Viewport
             * @instance
             */
            Viewport.prototype.x = 0;

            /**
             * Viewport y.
             * @member {number} y
             * @memberof flowcraft.v1.Viewport
             * @instance
             */
            Viewport.prototype.y = 0;

            /**
             * Viewport zoom.
             * @member {number} zoom
             * @memberof flowcraft.v1.Viewport
             * @instance
             */
            Viewport.prototype.zoom = 0;

            /**
             * Creates a new Viewport instance using the specified properties.
             * @function create
             * @memberof flowcraft.v1.Viewport
             * @static
             * @param {flowcraft.v1.IViewport=} [properties] Properties to set
             * @returns {flowcraft.v1.Viewport} Viewport instance
             */
            Viewport.create = function create(properties) {
                return new Viewport(properties);
            };

            /**
             * Encodes the specified Viewport message. Does not implicitly {@link flowcraft.v1.Viewport.verify|verify} messages.
             * @function encode
             * @memberof flowcraft.v1.Viewport
             * @static
             * @param {flowcraft.v1.IViewport} message Viewport message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            Viewport.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.x != null && Object.hasOwnProperty.call(message, "x"))
                    writer.uint32(/* id 1, wireType 1 =*/9).double(message.x);
                if (message.y != null && Object.hasOwnProperty.call(message, "y"))
                    writer.uint32(/* id 2, wireType 1 =*/17).double(message.y);
                if (message.zoom != null && Object.hasOwnProperty.call(message, "zoom"))
                    writer.uint32(/* id 3, wireType 1 =*/25).double(message.zoom);
                return writer;
            };

            /**
             * Encodes the specified Viewport message, length delimited. Does not implicitly {@link flowcraft.v1.Viewport.verify|verify} messages.
             * @function encodeDelimited
             * @memberof flowcraft.v1.Viewport
             * @static
             * @param {flowcraft.v1.IViewport} message Viewport message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            Viewport.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes a Viewport message from the specified reader or buffer.
             * @function decode
             * @memberof flowcraft.v1.Viewport
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {flowcraft.v1.Viewport} Viewport
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            Viewport.decode = function decode(reader, length, error) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.flowcraft.v1.Viewport();
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    if (tag === error)
                        break;
                    switch (tag >>> 3) {
                    case 1: {
                            message.x = reader.double();
                            break;
                        }
                    case 2: {
                            message.y = reader.double();
                            break;
                        }
                    case 3: {
                            message.zoom = reader.double();
                            break;
                        }
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };

            /**
             * Decodes a Viewport message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof flowcraft.v1.Viewport
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {flowcraft.v1.Viewport} Viewport
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            Viewport.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies a Viewport message.
             * @function verify
             * @memberof flowcraft.v1.Viewport
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            Viewport.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.x != null && message.hasOwnProperty("x"))
                    if (typeof message.x !== "number")
                        return "x: number expected";
                if (message.y != null && message.hasOwnProperty("y"))
                    if (typeof message.y !== "number")
                        return "y: number expected";
                if (message.zoom != null && message.hasOwnProperty("zoom"))
                    if (typeof message.zoom !== "number")
                        return "zoom: number expected";
                return null;
            };

            /**
             * Creates a Viewport message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof flowcraft.v1.Viewport
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {flowcraft.v1.Viewport} Viewport
             */
            Viewport.fromObject = function fromObject(object) {
                if (object instanceof $root.flowcraft.v1.Viewport)
                    return object;
                let message = new $root.flowcraft.v1.Viewport();
                if (object.x != null)
                    message.x = Number(object.x);
                if (object.y != null)
                    message.y = Number(object.y);
                if (object.zoom != null)
                    message.zoom = Number(object.zoom);
                return message;
            };

            /**
             * Creates a plain object from a Viewport message. Also converts values to other types if specified.
             * @function toObject
             * @memberof flowcraft.v1.Viewport
             * @static
             * @param {flowcraft.v1.Viewport} message Viewport
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            Viewport.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                let object = {};
                if (options.defaults) {
                    object.x = 0;
                    object.y = 0;
                    object.zoom = 0;
                }
                if (message.x != null && message.hasOwnProperty("x"))
                    object.x = options.json && !isFinite(message.x) ? String(message.x) : message.x;
                if (message.y != null && message.hasOwnProperty("y"))
                    object.y = options.json && !isFinite(message.y) ? String(message.y) : message.y;
                if (message.zoom != null && message.hasOwnProperty("zoom"))
                    object.zoom = options.json && !isFinite(message.zoom) ? String(message.zoom) : message.zoom;
                return object;
            };

            /**
             * Converts this Viewport to JSON.
             * @function toJSON
             * @memberof flowcraft.v1.Viewport
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            Viewport.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };

            /**
             * Gets the default type url for Viewport
             * @function getTypeUrl
             * @memberof flowcraft.v1.Viewport
             * @static
             * @param {string} [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns {string} The default type url
             */
            Viewport.getTypeUrl = function getTypeUrl(typeUrlPrefix) {
                if (typeUrlPrefix === undefined) {
                    typeUrlPrefix = "type.googleapis.com";
                }
                return typeUrlPrefix + "/flowcraft.v1.Viewport";
            };

            return Viewport;
        })();

        /**
         * PortStyle enum.
         * @name flowcraft.v1.PortStyle
         * @enum {number}
         * @property {number} PORT_STYLE_CIRCLE=0 PORT_STYLE_CIRCLE value
         * @property {number} PORT_STYLE_SQUARE=1 PORT_STYLE_SQUARE value
         * @property {number} PORT_STYLE_DIAMOND=2 PORT_STYLE_DIAMOND value
         * @property {number} PORT_STYLE_DASH=3 PORT_STYLE_DASH value
         */
        v1.PortStyle = (function() {
            const valuesById = {}, values = Object.create(valuesById);
            values[valuesById[0] = "PORT_STYLE_CIRCLE"] = 0;
            values[valuesById[1] = "PORT_STYLE_SQUARE"] = 1;
            values[valuesById[2] = "PORT_STYLE_DIAMOND"] = 2;
            values[valuesById[3] = "PORT_STYLE_DASH"] = 3;
            return values;
        })();

        v1.PortType = (function() {

            /**
             * Properties of a PortType.
             * @memberof flowcraft.v1
             * @interface IPortType
             * @property {string|null} [mainType] PortType mainType
             * @property {string|null} [itemType] PortType itemType
             * @property {boolean|null} [isGeneric] PortType isGeneric
             */

            /**
             * Constructs a new PortType.
             * @memberof flowcraft.v1
             * @classdesc Represents a PortType.
             * @implements IPortType
             * @constructor
             * @param {flowcraft.v1.IPortType=} [properties] Properties to set
             */
            function PortType(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * PortType mainType.
             * @member {string} mainType
             * @memberof flowcraft.v1.PortType
             * @instance
             */
            PortType.prototype.mainType = "";

            /**
             * PortType itemType.
             * @member {string} itemType
             * @memberof flowcraft.v1.PortType
             * @instance
             */
            PortType.prototype.itemType = "";

            /**
             * PortType isGeneric.
             * @member {boolean} isGeneric
             * @memberof flowcraft.v1.PortType
             * @instance
             */
            PortType.prototype.isGeneric = false;

            /**
             * Creates a new PortType instance using the specified properties.
             * @function create
             * @memberof flowcraft.v1.PortType
             * @static
             * @param {flowcraft.v1.IPortType=} [properties] Properties to set
             * @returns {flowcraft.v1.PortType} PortType instance
             */
            PortType.create = function create(properties) {
                return new PortType(properties);
            };

            /**
             * Encodes the specified PortType message. Does not implicitly {@link flowcraft.v1.PortType.verify|verify} messages.
             * @function encode
             * @memberof flowcraft.v1.PortType
             * @static
             * @param {flowcraft.v1.IPortType} message PortType message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            PortType.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.mainType != null && Object.hasOwnProperty.call(message, "mainType"))
                    writer.uint32(/* id 1, wireType 2 =*/10).string(message.mainType);
                if (message.itemType != null && Object.hasOwnProperty.call(message, "itemType"))
                    writer.uint32(/* id 2, wireType 2 =*/18).string(message.itemType);
                if (message.isGeneric != null && Object.hasOwnProperty.call(message, "isGeneric"))
                    writer.uint32(/* id 3, wireType 0 =*/24).bool(message.isGeneric);
                return writer;
            };

            /**
             * Encodes the specified PortType message, length delimited. Does not implicitly {@link flowcraft.v1.PortType.verify|verify} messages.
             * @function encodeDelimited
             * @memberof flowcraft.v1.PortType
             * @static
             * @param {flowcraft.v1.IPortType} message PortType message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            PortType.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes a PortType message from the specified reader or buffer.
             * @function decode
             * @memberof flowcraft.v1.PortType
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {flowcraft.v1.PortType} PortType
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            PortType.decode = function decode(reader, length, error) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.flowcraft.v1.PortType();
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    if (tag === error)
                        break;
                    switch (tag >>> 3) {
                    case 1: {
                            message.mainType = reader.string();
                            break;
                        }
                    case 2: {
                            message.itemType = reader.string();
                            break;
                        }
                    case 3: {
                            message.isGeneric = reader.bool();
                            break;
                        }
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };

            /**
             * Decodes a PortType message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof flowcraft.v1.PortType
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {flowcraft.v1.PortType} PortType
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            PortType.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies a PortType message.
             * @function verify
             * @memberof flowcraft.v1.PortType
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            PortType.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.mainType != null && message.hasOwnProperty("mainType"))
                    if (!$util.isString(message.mainType))
                        return "mainType: string expected";
                if (message.itemType != null && message.hasOwnProperty("itemType"))
                    if (!$util.isString(message.itemType))
                        return "itemType: string expected";
                if (message.isGeneric != null && message.hasOwnProperty("isGeneric"))
                    if (typeof message.isGeneric !== "boolean")
                        return "isGeneric: boolean expected";
                return null;
            };

            /**
             * Creates a PortType message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof flowcraft.v1.PortType
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {flowcraft.v1.PortType} PortType
             */
            PortType.fromObject = function fromObject(object) {
                if (object instanceof $root.flowcraft.v1.PortType)
                    return object;
                let message = new $root.flowcraft.v1.PortType();
                if (object.mainType != null)
                    message.mainType = String(object.mainType);
                if (object.itemType != null)
                    message.itemType = String(object.itemType);
                if (object.isGeneric != null)
                    message.isGeneric = Boolean(object.isGeneric);
                return message;
            };

            /**
             * Creates a plain object from a PortType message. Also converts values to other types if specified.
             * @function toObject
             * @memberof flowcraft.v1.PortType
             * @static
             * @param {flowcraft.v1.PortType} message PortType
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            PortType.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                let object = {};
                if (options.defaults) {
                    object.mainType = "";
                    object.itemType = "";
                    object.isGeneric = false;
                }
                if (message.mainType != null && message.hasOwnProperty("mainType"))
                    object.mainType = message.mainType;
                if (message.itemType != null && message.hasOwnProperty("itemType"))
                    object.itemType = message.itemType;
                if (message.isGeneric != null && message.hasOwnProperty("isGeneric"))
                    object.isGeneric = message.isGeneric;
                return object;
            };

            /**
             * Converts this PortType to JSON.
             * @function toJSON
             * @memberof flowcraft.v1.PortType
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            PortType.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };

            /**
             * Gets the default type url for PortType
             * @function getTypeUrl
             * @memberof flowcraft.v1.PortType
             * @static
             * @param {string} [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns {string} The default type url
             */
            PortType.getTypeUrl = function getTypeUrl(typeUrlPrefix) {
                if (typeUrlPrefix === undefined) {
                    typeUrlPrefix = "type.googleapis.com";
                }
                return typeUrlPrefix + "/flowcraft.v1.PortType";
            };

            return PortType;
        })();

        v1.Port = (function() {

            /**
             * Properties of a Port.
             * @memberof flowcraft.v1
             * @interface IPort
             * @property {string|null} [id] Port id
             * @property {string|null} [label] Port label
             * @property {flowcraft.v1.IPortType|null} [type] Port type
             * @property {flowcraft.v1.PortStyle|null} [style] Port style
             * @property {string|null} [color] Port color
             * @property {string|null} [description] Port description
             */

            /**
             * Constructs a new Port.
             * @memberof flowcraft.v1
             * @classdesc Represents a Port.
             * @implements IPort
             * @constructor
             * @param {flowcraft.v1.IPort=} [properties] Properties to set
             */
            function Port(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * Port id.
             * @member {string} id
             * @memberof flowcraft.v1.Port
             * @instance
             */
            Port.prototype.id = "";

            /**
             * Port label.
             * @member {string} label
             * @memberof flowcraft.v1.Port
             * @instance
             */
            Port.prototype.label = "";

            /**
             * Port type.
             * @member {flowcraft.v1.IPortType|null|undefined} type
             * @memberof flowcraft.v1.Port
             * @instance
             */
            Port.prototype.type = null;

            /**
             * Port style.
             * @member {flowcraft.v1.PortStyle} style
             * @memberof flowcraft.v1.Port
             * @instance
             */
            Port.prototype.style = 0;

            /**
             * Port color.
             * @member {string} color
             * @memberof flowcraft.v1.Port
             * @instance
             */
            Port.prototype.color = "";

            /**
             * Port description.
             * @member {string} description
             * @memberof flowcraft.v1.Port
             * @instance
             */
            Port.prototype.description = "";

            /**
             * Creates a new Port instance using the specified properties.
             * @function create
             * @memberof flowcraft.v1.Port
             * @static
             * @param {flowcraft.v1.IPort=} [properties] Properties to set
             * @returns {flowcraft.v1.Port} Port instance
             */
            Port.create = function create(properties) {
                return new Port(properties);
            };

            /**
             * Encodes the specified Port message. Does not implicitly {@link flowcraft.v1.Port.verify|verify} messages.
             * @function encode
             * @memberof flowcraft.v1.Port
             * @static
             * @param {flowcraft.v1.IPort} message Port message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            Port.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.id != null && Object.hasOwnProperty.call(message, "id"))
                    writer.uint32(/* id 1, wireType 2 =*/10).string(message.id);
                if (message.label != null && Object.hasOwnProperty.call(message, "label"))
                    writer.uint32(/* id 2, wireType 2 =*/18).string(message.label);
                if (message.type != null && Object.hasOwnProperty.call(message, "type"))
                    $root.flowcraft.v1.PortType.encode(message.type, writer.uint32(/* id 3, wireType 2 =*/26).fork()).ldelim();
                if (message.style != null && Object.hasOwnProperty.call(message, "style"))
                    writer.uint32(/* id 4, wireType 0 =*/32).int32(message.style);
                if (message.color != null && Object.hasOwnProperty.call(message, "color"))
                    writer.uint32(/* id 5, wireType 2 =*/42).string(message.color);
                if (message.description != null && Object.hasOwnProperty.call(message, "description"))
                    writer.uint32(/* id 6, wireType 2 =*/50).string(message.description);
                return writer;
            };

            /**
             * Encodes the specified Port message, length delimited. Does not implicitly {@link flowcraft.v1.Port.verify|verify} messages.
             * @function encodeDelimited
             * @memberof flowcraft.v1.Port
             * @static
             * @param {flowcraft.v1.IPort} message Port message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            Port.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes a Port message from the specified reader or buffer.
             * @function decode
             * @memberof flowcraft.v1.Port
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {flowcraft.v1.Port} Port
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            Port.decode = function decode(reader, length, error) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.flowcraft.v1.Port();
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    if (tag === error)
                        break;
                    switch (tag >>> 3) {
                    case 1: {
                            message.id = reader.string();
                            break;
                        }
                    case 2: {
                            message.label = reader.string();
                            break;
                        }
                    case 3: {
                            message.type = $root.flowcraft.v1.PortType.decode(reader, reader.uint32());
                            break;
                        }
                    case 4: {
                            message.style = reader.int32();
                            break;
                        }
                    case 5: {
                            message.color = reader.string();
                            break;
                        }
                    case 6: {
                            message.description = reader.string();
                            break;
                        }
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };

            /**
             * Decodes a Port message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof flowcraft.v1.Port
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {flowcraft.v1.Port} Port
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            Port.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies a Port message.
             * @function verify
             * @memberof flowcraft.v1.Port
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            Port.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.id != null && message.hasOwnProperty("id"))
                    if (!$util.isString(message.id))
                        return "id: string expected";
                if (message.label != null && message.hasOwnProperty("label"))
                    if (!$util.isString(message.label))
                        return "label: string expected";
                if (message.type != null && message.hasOwnProperty("type")) {
                    let error = $root.flowcraft.v1.PortType.verify(message.type);
                    if (error)
                        return "type." + error;
                }
                if (message.style != null && message.hasOwnProperty("style"))
                    switch (message.style) {
                    default:
                        return "style: enum value expected";
                    case 0:
                    case 1:
                    case 2:
                    case 3:
                        break;
                    }
                if (message.color != null && message.hasOwnProperty("color"))
                    if (!$util.isString(message.color))
                        return "color: string expected";
                if (message.description != null && message.hasOwnProperty("description"))
                    if (!$util.isString(message.description))
                        return "description: string expected";
                return null;
            };

            /**
             * Creates a Port message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof flowcraft.v1.Port
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {flowcraft.v1.Port} Port
             */
            Port.fromObject = function fromObject(object) {
                if (object instanceof $root.flowcraft.v1.Port)
                    return object;
                let message = new $root.flowcraft.v1.Port();
                if (object.id != null)
                    message.id = String(object.id);
                if (object.label != null)
                    message.label = String(object.label);
                if (object.type != null) {
                    if (typeof object.type !== "object")
                        throw TypeError(".flowcraft.v1.Port.type: object expected");
                    message.type = $root.flowcraft.v1.PortType.fromObject(object.type);
                }
                switch (object.style) {
                default:
                    if (typeof object.style === "number") {
                        message.style = object.style;
                        break;
                    }
                    break;
                case "PORT_STYLE_CIRCLE":
                case 0:
                    message.style = 0;
                    break;
                case "PORT_STYLE_SQUARE":
                case 1:
                    message.style = 1;
                    break;
                case "PORT_STYLE_DIAMOND":
                case 2:
                    message.style = 2;
                    break;
                case "PORT_STYLE_DASH":
                case 3:
                    message.style = 3;
                    break;
                }
                if (object.color != null)
                    message.color = String(object.color);
                if (object.description != null)
                    message.description = String(object.description);
                return message;
            };

            /**
             * Creates a plain object from a Port message. Also converts values to other types if specified.
             * @function toObject
             * @memberof flowcraft.v1.Port
             * @static
             * @param {flowcraft.v1.Port} message Port
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            Port.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                let object = {};
                if (options.defaults) {
                    object.id = "";
                    object.label = "";
                    object.type = null;
                    object.style = options.enums === String ? "PORT_STYLE_CIRCLE" : 0;
                    object.color = "";
                    object.description = "";
                }
                if (message.id != null && message.hasOwnProperty("id"))
                    object.id = message.id;
                if (message.label != null && message.hasOwnProperty("label"))
                    object.label = message.label;
                if (message.type != null && message.hasOwnProperty("type"))
                    object.type = $root.flowcraft.v1.PortType.toObject(message.type, options);
                if (message.style != null && message.hasOwnProperty("style"))
                    object.style = options.enums === String ? $root.flowcraft.v1.PortStyle[message.style] === undefined ? message.style : $root.flowcraft.v1.PortStyle[message.style] : message.style;
                if (message.color != null && message.hasOwnProperty("color"))
                    object.color = message.color;
                if (message.description != null && message.hasOwnProperty("description"))
                    object.description = message.description;
                return object;
            };

            /**
             * Converts this Port to JSON.
             * @function toJSON
             * @memberof flowcraft.v1.Port
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            Port.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };

            /**
             * Gets the default type url for Port
             * @function getTypeUrl
             * @memberof flowcraft.v1.Port
             * @static
             * @param {string} [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns {string} The default type url
             */
            Port.getTypeUrl = function getTypeUrl(typeUrlPrefix) {
                if (typeUrlPrefix === undefined) {
                    typeUrlPrefix = "type.googleapis.com";
                }
                return typeUrlPrefix + "/flowcraft.v1.Port";
            };

            return Port;
        })();

        v1.Edge = (function() {

            /**
             * Properties of an Edge.
             * @memberof flowcraft.v1
             * @interface IEdge
             * @property {string|null} [id] Edge id
             * @property {string|null} [source] Edge source
             * @property {string|null} [target] Edge target
             * @property {string|null} [sourceHandle] Edge sourceHandle
             * @property {string|null} [targetHandle] Edge targetHandle
             * @property {Object.<string,string>|null} [metadata] Edge metadata
             */

            /**
             * Constructs a new Edge.
             * @memberof flowcraft.v1
             * @classdesc Represents an Edge.
             * @implements IEdge
             * @constructor
             * @param {flowcraft.v1.IEdge=} [properties] Properties to set
             */
            function Edge(properties) {
                this.metadata = {};
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * Edge id.
             * @member {string} id
             * @memberof flowcraft.v1.Edge
             * @instance
             */
            Edge.prototype.id = "";

            /**
             * Edge source.
             * @member {string} source
             * @memberof flowcraft.v1.Edge
             * @instance
             */
            Edge.prototype.source = "";

            /**
             * Edge target.
             * @member {string} target
             * @memberof flowcraft.v1.Edge
             * @instance
             */
            Edge.prototype.target = "";

            /**
             * Edge sourceHandle.
             * @member {string} sourceHandle
             * @memberof flowcraft.v1.Edge
             * @instance
             */
            Edge.prototype.sourceHandle = "";

            /**
             * Edge targetHandle.
             * @member {string} targetHandle
             * @memberof flowcraft.v1.Edge
             * @instance
             */
            Edge.prototype.targetHandle = "";

            /**
             * Edge metadata.
             * @member {Object.<string,string>} metadata
             * @memberof flowcraft.v1.Edge
             * @instance
             */
            Edge.prototype.metadata = $util.emptyObject;

            /**
             * Creates a new Edge instance using the specified properties.
             * @function create
             * @memberof flowcraft.v1.Edge
             * @static
             * @param {flowcraft.v1.IEdge=} [properties] Properties to set
             * @returns {flowcraft.v1.Edge} Edge instance
             */
            Edge.create = function create(properties) {
                return new Edge(properties);
            };

            /**
             * Encodes the specified Edge message. Does not implicitly {@link flowcraft.v1.Edge.verify|verify} messages.
             * @function encode
             * @memberof flowcraft.v1.Edge
             * @static
             * @param {flowcraft.v1.IEdge} message Edge message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            Edge.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.id != null && Object.hasOwnProperty.call(message, "id"))
                    writer.uint32(/* id 1, wireType 2 =*/10).string(message.id);
                if (message.source != null && Object.hasOwnProperty.call(message, "source"))
                    writer.uint32(/* id 2, wireType 2 =*/18).string(message.source);
                if (message.target != null && Object.hasOwnProperty.call(message, "target"))
                    writer.uint32(/* id 3, wireType 2 =*/26).string(message.target);
                if (message.sourceHandle != null && Object.hasOwnProperty.call(message, "sourceHandle"))
                    writer.uint32(/* id 4, wireType 2 =*/34).string(message.sourceHandle);
                if (message.targetHandle != null && Object.hasOwnProperty.call(message, "targetHandle"))
                    writer.uint32(/* id 5, wireType 2 =*/42).string(message.targetHandle);
                if (message.metadata != null && Object.hasOwnProperty.call(message, "metadata"))
                    for (let keys = Object.keys(message.metadata), i = 0; i < keys.length; ++i)
                        writer.uint32(/* id 6, wireType 2 =*/50).fork().uint32(/* id 1, wireType 2 =*/10).string(keys[i]).uint32(/* id 2, wireType 2 =*/18).string(message.metadata[keys[i]]).ldelim();
                return writer;
            };

            /**
             * Encodes the specified Edge message, length delimited. Does not implicitly {@link flowcraft.v1.Edge.verify|verify} messages.
             * @function encodeDelimited
             * @memberof flowcraft.v1.Edge
             * @static
             * @param {flowcraft.v1.IEdge} message Edge message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            Edge.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes an Edge message from the specified reader or buffer.
             * @function decode
             * @memberof flowcraft.v1.Edge
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {flowcraft.v1.Edge} Edge
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            Edge.decode = function decode(reader, length, error) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.flowcraft.v1.Edge(), key, value;
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    if (tag === error)
                        break;
                    switch (tag >>> 3) {
                    case 1: {
                            message.id = reader.string();
                            break;
                        }
                    case 2: {
                            message.source = reader.string();
                            break;
                        }
                    case 3: {
                            message.target = reader.string();
                            break;
                        }
                    case 4: {
                            message.sourceHandle = reader.string();
                            break;
                        }
                    case 5: {
                            message.targetHandle = reader.string();
                            break;
                        }
                    case 6: {
                            if (message.metadata === $util.emptyObject)
                                message.metadata = {};
                            let end2 = reader.uint32() + reader.pos;
                            key = "";
                            value = "";
                            while (reader.pos < end2) {
                                let tag2 = reader.uint32();
                                switch (tag2 >>> 3) {
                                case 1:
                                    key = reader.string();
                                    break;
                                case 2:
                                    value = reader.string();
                                    break;
                                default:
                                    reader.skipType(tag2 & 7);
                                    break;
                                }
                            }
                            message.metadata[key] = value;
                            break;
                        }
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };

            /**
             * Decodes an Edge message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof flowcraft.v1.Edge
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {flowcraft.v1.Edge} Edge
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            Edge.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies an Edge message.
             * @function verify
             * @memberof flowcraft.v1.Edge
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            Edge.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.id != null && message.hasOwnProperty("id"))
                    if (!$util.isString(message.id))
                        return "id: string expected";
                if (message.source != null && message.hasOwnProperty("source"))
                    if (!$util.isString(message.source))
                        return "source: string expected";
                if (message.target != null && message.hasOwnProperty("target"))
                    if (!$util.isString(message.target))
                        return "target: string expected";
                if (message.sourceHandle != null && message.hasOwnProperty("sourceHandle"))
                    if (!$util.isString(message.sourceHandle))
                        return "sourceHandle: string expected";
                if (message.targetHandle != null && message.hasOwnProperty("targetHandle"))
                    if (!$util.isString(message.targetHandle))
                        return "targetHandle: string expected";
                if (message.metadata != null && message.hasOwnProperty("metadata")) {
                    if (!$util.isObject(message.metadata))
                        return "metadata: object expected";
                    let key = Object.keys(message.metadata);
                    for (let i = 0; i < key.length; ++i)
                        if (!$util.isString(message.metadata[key[i]]))
                            return "metadata: string{k:string} expected";
                }
                return null;
            };

            /**
             * Creates an Edge message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof flowcraft.v1.Edge
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {flowcraft.v1.Edge} Edge
             */
            Edge.fromObject = function fromObject(object) {
                if (object instanceof $root.flowcraft.v1.Edge)
                    return object;
                let message = new $root.flowcraft.v1.Edge();
                if (object.id != null)
                    message.id = String(object.id);
                if (object.source != null)
                    message.source = String(object.source);
                if (object.target != null)
                    message.target = String(object.target);
                if (object.sourceHandle != null)
                    message.sourceHandle = String(object.sourceHandle);
                if (object.targetHandle != null)
                    message.targetHandle = String(object.targetHandle);
                if (object.metadata) {
                    if (typeof object.metadata !== "object")
                        throw TypeError(".flowcraft.v1.Edge.metadata: object expected");
                    message.metadata = {};
                    for (let keys = Object.keys(object.metadata), i = 0; i < keys.length; ++i)
                        message.metadata[keys[i]] = String(object.metadata[keys[i]]);
                }
                return message;
            };

            /**
             * Creates a plain object from an Edge message. Also converts values to other types if specified.
             * @function toObject
             * @memberof flowcraft.v1.Edge
             * @static
             * @param {flowcraft.v1.Edge} message Edge
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            Edge.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                let object = {};
                if (options.objects || options.defaults)
                    object.metadata = {};
                if (options.defaults) {
                    object.id = "";
                    object.source = "";
                    object.target = "";
                    object.sourceHandle = "";
                    object.targetHandle = "";
                }
                if (message.id != null && message.hasOwnProperty("id"))
                    object.id = message.id;
                if (message.source != null && message.hasOwnProperty("source"))
                    object.source = message.source;
                if (message.target != null && message.hasOwnProperty("target"))
                    object.target = message.target;
                if (message.sourceHandle != null && message.hasOwnProperty("sourceHandle"))
                    object.sourceHandle = message.sourceHandle;
                if (message.targetHandle != null && message.hasOwnProperty("targetHandle"))
                    object.targetHandle = message.targetHandle;
                let keys2;
                if (message.metadata && (keys2 = Object.keys(message.metadata)).length) {
                    object.metadata = {};
                    for (let j = 0; j < keys2.length; ++j)
                        object.metadata[keys2[j]] = message.metadata[keys2[j]];
                }
                return object;
            };

            /**
             * Converts this Edge to JSON.
             * @function toJSON
             * @memberof flowcraft.v1.Edge
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            Edge.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };

            /**
             * Gets the default type url for Edge
             * @function getTypeUrl
             * @memberof flowcraft.v1.Edge
             * @static
             * @param {string} [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns {string} The default type url
             */
            Edge.getTypeUrl = function getTypeUrl(typeUrlPrefix) {
                if (typeUrlPrefix === undefined) {
                    typeUrlPrefix = "type.googleapis.com";
                }
                return typeUrlPrefix + "/flowcraft.v1.Edge";
            };

            return Edge;
        })();

        v1.FlowMessage = (function() {

            /**
             * Properties of a FlowMessage.
             * @memberof flowcraft.v1
             * @interface IFlowMessage
             * @property {string|null} [messageId] FlowMessage messageId
             * @property {number|Long|null} [timestamp] FlowMessage timestamp
             * @property {flowcraft.v1.ISyncRequest|null} [syncRequest] FlowMessage syncRequest
             * @property {Uint8Array|null} [yjsUpdate] FlowMessage yjsUpdate
             * @property {flowcraft.v1.IUpdateNodeRequest|null} [nodeUpdate] FlowMessage nodeUpdate
             * @property {flowcraft.v1.IUpdateWidgetRequest|null} [widgetUpdate] FlowMessage widgetUpdate
             * @property {flowcraft.v1.IActionExecutionRequest|null} [actionExecute] FlowMessage actionExecute
             * @property {flowcraft.v1.ITaskCancelRequest|null} [taskCancel] FlowMessage taskCancel
             * @property {flowcraft.v1.IViewportUpdate|null} [viewportUpdate] FlowMessage viewportUpdate
             * @property {flowcraft.v1.IGraphSnapshot|null} [snapshot] FlowMessage snapshot
             * @property {flowcraft.v1.IMutationList|null} [mutations] FlowMessage mutations
             * @property {flowcraft.v1.ITaskUpdate|null} [taskUpdate] FlowMessage taskUpdate
             * @property {flowcraft.v1.IStreamChunk|null} [streamChunk] FlowMessage streamChunk
             * @property {flowcraft.v1.IErrorResponse|null} [error] FlowMessage error
             */

            /**
             * Constructs a new FlowMessage.
             * @memberof flowcraft.v1
             * @classdesc Represents a FlowMessage.
             * @implements IFlowMessage
             * @constructor
             * @param {flowcraft.v1.IFlowMessage=} [properties] Properties to set
             */
            function FlowMessage(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * FlowMessage messageId.
             * @member {string} messageId
             * @memberof flowcraft.v1.FlowMessage
             * @instance
             */
            FlowMessage.prototype.messageId = "";

            /**
             * FlowMessage timestamp.
             * @member {number|Long} timestamp
             * @memberof flowcraft.v1.FlowMessage
             * @instance
             */
            FlowMessage.prototype.timestamp = $util.Long ? $util.Long.fromBits(0,0,false) : 0;

            /**
             * FlowMessage syncRequest.
             * @member {flowcraft.v1.ISyncRequest|null|undefined} syncRequest
             * @memberof flowcraft.v1.FlowMessage
             * @instance
             */
            FlowMessage.prototype.syncRequest = null;

            /**
             * FlowMessage yjsUpdate.
             * @member {Uint8Array|null|undefined} yjsUpdate
             * @memberof flowcraft.v1.FlowMessage
             * @instance
             */
            FlowMessage.prototype.yjsUpdate = null;

            /**
             * FlowMessage nodeUpdate.
             * @member {flowcraft.v1.IUpdateNodeRequest|null|undefined} nodeUpdate
             * @memberof flowcraft.v1.FlowMessage
             * @instance
             */
            FlowMessage.prototype.nodeUpdate = null;

            /**
             * FlowMessage widgetUpdate.
             * @member {flowcraft.v1.IUpdateWidgetRequest|null|undefined} widgetUpdate
             * @memberof flowcraft.v1.FlowMessage
             * @instance
             */
            FlowMessage.prototype.widgetUpdate = null;

            /**
             * FlowMessage actionExecute.
             * @member {flowcraft.v1.IActionExecutionRequest|null|undefined} actionExecute
             * @memberof flowcraft.v1.FlowMessage
             * @instance
             */
            FlowMessage.prototype.actionExecute = null;

            /**
             * FlowMessage taskCancel.
             * @member {flowcraft.v1.ITaskCancelRequest|null|undefined} taskCancel
             * @memberof flowcraft.v1.FlowMessage
             * @instance
             */
            FlowMessage.prototype.taskCancel = null;

            /**
             * FlowMessage viewportUpdate.
             * @member {flowcraft.v1.IViewportUpdate|null|undefined} viewportUpdate
             * @memberof flowcraft.v1.FlowMessage
             * @instance
             */
            FlowMessage.prototype.viewportUpdate = null;

            /**
             * FlowMessage snapshot.
             * @member {flowcraft.v1.IGraphSnapshot|null|undefined} snapshot
             * @memberof flowcraft.v1.FlowMessage
             * @instance
             */
            FlowMessage.prototype.snapshot = null;

            /**
             * FlowMessage mutations.
             * @member {flowcraft.v1.IMutationList|null|undefined} mutations
             * @memberof flowcraft.v1.FlowMessage
             * @instance
             */
            FlowMessage.prototype.mutations = null;

            /**
             * FlowMessage taskUpdate.
             * @member {flowcraft.v1.ITaskUpdate|null|undefined} taskUpdate
             * @memberof flowcraft.v1.FlowMessage
             * @instance
             */
            FlowMessage.prototype.taskUpdate = null;

            /**
             * FlowMessage streamChunk.
             * @member {flowcraft.v1.IStreamChunk|null|undefined} streamChunk
             * @memberof flowcraft.v1.FlowMessage
             * @instance
             */
            FlowMessage.prototype.streamChunk = null;

            /**
             * FlowMessage error.
             * @member {flowcraft.v1.IErrorResponse|null|undefined} error
             * @memberof flowcraft.v1.FlowMessage
             * @instance
             */
            FlowMessage.prototype.error = null;

            // OneOf field names bound to virtual getters and setters
            let $oneOfFields;

            /**
             * FlowMessage payload.
             * @member {"syncRequest"|"yjsUpdate"|"nodeUpdate"|"widgetUpdate"|"actionExecute"|"taskCancel"|"viewportUpdate"|"snapshot"|"mutations"|"taskUpdate"|"streamChunk"|"error"|undefined} payload
             * @memberof flowcraft.v1.FlowMessage
             * @instance
             */
            Object.defineProperty(FlowMessage.prototype, "payload", {
                get: $util.oneOfGetter($oneOfFields = ["syncRequest", "yjsUpdate", "nodeUpdate", "widgetUpdate", "actionExecute", "taskCancel", "viewportUpdate", "snapshot", "mutations", "taskUpdate", "streamChunk", "error"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * Creates a new FlowMessage instance using the specified properties.
             * @function create
             * @memberof flowcraft.v1.FlowMessage
             * @static
             * @param {flowcraft.v1.IFlowMessage=} [properties] Properties to set
             * @returns {flowcraft.v1.FlowMessage} FlowMessage instance
             */
            FlowMessage.create = function create(properties) {
                return new FlowMessage(properties);
            };

            /**
             * Encodes the specified FlowMessage message. Does not implicitly {@link flowcraft.v1.FlowMessage.verify|verify} messages.
             * @function encode
             * @memberof flowcraft.v1.FlowMessage
             * @static
             * @param {flowcraft.v1.IFlowMessage} message FlowMessage message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            FlowMessage.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.messageId != null && Object.hasOwnProperty.call(message, "messageId"))
                    writer.uint32(/* id 1, wireType 2 =*/10).string(message.messageId);
                if (message.timestamp != null && Object.hasOwnProperty.call(message, "timestamp"))
                    writer.uint32(/* id 2, wireType 0 =*/16).int64(message.timestamp);
                if (message.syncRequest != null && Object.hasOwnProperty.call(message, "syncRequest"))
                    $root.flowcraft.v1.SyncRequest.encode(message.syncRequest, writer.uint32(/* id 10, wireType 2 =*/82).fork()).ldelim();
                if (message.yjsUpdate != null && Object.hasOwnProperty.call(message, "yjsUpdate"))
                    writer.uint32(/* id 11, wireType 2 =*/90).bytes(message.yjsUpdate);
                if (message.nodeUpdate != null && Object.hasOwnProperty.call(message, "nodeUpdate"))
                    $root.flowcraft.v1.UpdateNodeRequest.encode(message.nodeUpdate, writer.uint32(/* id 12, wireType 2 =*/98).fork()).ldelim();
                if (message.widgetUpdate != null && Object.hasOwnProperty.call(message, "widgetUpdate"))
                    $root.flowcraft.v1.UpdateWidgetRequest.encode(message.widgetUpdate, writer.uint32(/* id 13, wireType 2 =*/106).fork()).ldelim();
                if (message.actionExecute != null && Object.hasOwnProperty.call(message, "actionExecute"))
                    $root.flowcraft.v1.ActionExecutionRequest.encode(message.actionExecute, writer.uint32(/* id 14, wireType 2 =*/114).fork()).ldelim();
                if (message.taskCancel != null && Object.hasOwnProperty.call(message, "taskCancel"))
                    $root.flowcraft.v1.TaskCancelRequest.encode(message.taskCancel, writer.uint32(/* id 15, wireType 2 =*/122).fork()).ldelim();
                if (message.viewportUpdate != null && Object.hasOwnProperty.call(message, "viewportUpdate"))
                    $root.flowcraft.v1.ViewportUpdate.encode(message.viewportUpdate, writer.uint32(/* id 16, wireType 2 =*/130).fork()).ldelim();
                if (message.snapshot != null && Object.hasOwnProperty.call(message, "snapshot"))
                    $root.flowcraft.v1.GraphSnapshot.encode(message.snapshot, writer.uint32(/* id 20, wireType 2 =*/162).fork()).ldelim();
                if (message.mutations != null && Object.hasOwnProperty.call(message, "mutations"))
                    $root.flowcraft.v1.MutationList.encode(message.mutations, writer.uint32(/* id 21, wireType 2 =*/170).fork()).ldelim();
                if (message.taskUpdate != null && Object.hasOwnProperty.call(message, "taskUpdate"))
                    $root.flowcraft.v1.TaskUpdate.encode(message.taskUpdate, writer.uint32(/* id 22, wireType 2 =*/178).fork()).ldelim();
                if (message.streamChunk != null && Object.hasOwnProperty.call(message, "streamChunk"))
                    $root.flowcraft.v1.StreamChunk.encode(message.streamChunk, writer.uint32(/* id 23, wireType 2 =*/186).fork()).ldelim();
                if (message.error != null && Object.hasOwnProperty.call(message, "error"))
                    $root.flowcraft.v1.ErrorResponse.encode(message.error, writer.uint32(/* id 24, wireType 2 =*/194).fork()).ldelim();
                return writer;
            };

            /**
             * Encodes the specified FlowMessage message, length delimited. Does not implicitly {@link flowcraft.v1.FlowMessage.verify|verify} messages.
             * @function encodeDelimited
             * @memberof flowcraft.v1.FlowMessage
             * @static
             * @param {flowcraft.v1.IFlowMessage} message FlowMessage message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            FlowMessage.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes a FlowMessage message from the specified reader or buffer.
             * @function decode
             * @memberof flowcraft.v1.FlowMessage
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {flowcraft.v1.FlowMessage} FlowMessage
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            FlowMessage.decode = function decode(reader, length, error) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.flowcraft.v1.FlowMessage();
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    if (tag === error)
                        break;
                    switch (tag >>> 3) {
                    case 1: {
                            message.messageId = reader.string();
                            break;
                        }
                    case 2: {
                            message.timestamp = reader.int64();
                            break;
                        }
                    case 10: {
                            message.syncRequest = $root.flowcraft.v1.SyncRequest.decode(reader, reader.uint32());
                            break;
                        }
                    case 11: {
                            message.yjsUpdate = reader.bytes();
                            break;
                        }
                    case 12: {
                            message.nodeUpdate = $root.flowcraft.v1.UpdateNodeRequest.decode(reader, reader.uint32());
                            break;
                        }
                    case 13: {
                            message.widgetUpdate = $root.flowcraft.v1.UpdateWidgetRequest.decode(reader, reader.uint32());
                            break;
                        }
                    case 14: {
                            message.actionExecute = $root.flowcraft.v1.ActionExecutionRequest.decode(reader, reader.uint32());
                            break;
                        }
                    case 15: {
                            message.taskCancel = $root.flowcraft.v1.TaskCancelRequest.decode(reader, reader.uint32());
                            break;
                        }
                    case 16: {
                            message.viewportUpdate = $root.flowcraft.v1.ViewportUpdate.decode(reader, reader.uint32());
                            break;
                        }
                    case 20: {
                            message.snapshot = $root.flowcraft.v1.GraphSnapshot.decode(reader, reader.uint32());
                            break;
                        }
                    case 21: {
                            message.mutations = $root.flowcraft.v1.MutationList.decode(reader, reader.uint32());
                            break;
                        }
                    case 22: {
                            message.taskUpdate = $root.flowcraft.v1.TaskUpdate.decode(reader, reader.uint32());
                            break;
                        }
                    case 23: {
                            message.streamChunk = $root.flowcraft.v1.StreamChunk.decode(reader, reader.uint32());
                            break;
                        }
                    case 24: {
                            message.error = $root.flowcraft.v1.ErrorResponse.decode(reader, reader.uint32());
                            break;
                        }
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };

            /**
             * Decodes a FlowMessage message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof flowcraft.v1.FlowMessage
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {flowcraft.v1.FlowMessage} FlowMessage
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            FlowMessage.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies a FlowMessage message.
             * @function verify
             * @memberof flowcraft.v1.FlowMessage
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            FlowMessage.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                let properties = {};
                if (message.messageId != null && message.hasOwnProperty("messageId"))
                    if (!$util.isString(message.messageId))
                        return "messageId: string expected";
                if (message.timestamp != null && message.hasOwnProperty("timestamp"))
                    if (!$util.isInteger(message.timestamp) && !(message.timestamp && $util.isInteger(message.timestamp.low) && $util.isInteger(message.timestamp.high)))
                        return "timestamp: integer|Long expected";
                if (message.syncRequest != null && message.hasOwnProperty("syncRequest")) {
                    properties.payload = 1;
                    {
                        let error = $root.flowcraft.v1.SyncRequest.verify(message.syncRequest);
                        if (error)
                            return "syncRequest." + error;
                    }
                }
                if (message.yjsUpdate != null && message.hasOwnProperty("yjsUpdate")) {
                    if (properties.payload === 1)
                        return "payload: multiple values";
                    properties.payload = 1;
                    if (!(message.yjsUpdate && typeof message.yjsUpdate.length === "number" || $util.isString(message.yjsUpdate)))
                        return "yjsUpdate: buffer expected";
                }
                if (message.nodeUpdate != null && message.hasOwnProperty("nodeUpdate")) {
                    if (properties.payload === 1)
                        return "payload: multiple values";
                    properties.payload = 1;
                    {
                        let error = $root.flowcraft.v1.UpdateNodeRequest.verify(message.nodeUpdate);
                        if (error)
                            return "nodeUpdate." + error;
                    }
                }
                if (message.widgetUpdate != null && message.hasOwnProperty("widgetUpdate")) {
                    if (properties.payload === 1)
                        return "payload: multiple values";
                    properties.payload = 1;
                    {
                        let error = $root.flowcraft.v1.UpdateWidgetRequest.verify(message.widgetUpdate);
                        if (error)
                            return "widgetUpdate." + error;
                    }
                }
                if (message.actionExecute != null && message.hasOwnProperty("actionExecute")) {
                    if (properties.payload === 1)
                        return "payload: multiple values";
                    properties.payload = 1;
                    {
                        let error = $root.flowcraft.v1.ActionExecutionRequest.verify(message.actionExecute);
                        if (error)
                            return "actionExecute." + error;
                    }
                }
                if (message.taskCancel != null && message.hasOwnProperty("taskCancel")) {
                    if (properties.payload === 1)
                        return "payload: multiple values";
                    properties.payload = 1;
                    {
                        let error = $root.flowcraft.v1.TaskCancelRequest.verify(message.taskCancel);
                        if (error)
                            return "taskCancel." + error;
                    }
                }
                if (message.viewportUpdate != null && message.hasOwnProperty("viewportUpdate")) {
                    if (properties.payload === 1)
                        return "payload: multiple values";
                    properties.payload = 1;
                    {
                        let error = $root.flowcraft.v1.ViewportUpdate.verify(message.viewportUpdate);
                        if (error)
                            return "viewportUpdate." + error;
                    }
                }
                if (message.snapshot != null && message.hasOwnProperty("snapshot")) {
                    if (properties.payload === 1)
                        return "payload: multiple values";
                    properties.payload = 1;
                    {
                        let error = $root.flowcraft.v1.GraphSnapshot.verify(message.snapshot);
                        if (error)
                            return "snapshot." + error;
                    }
                }
                if (message.mutations != null && message.hasOwnProperty("mutations")) {
                    if (properties.payload === 1)
                        return "payload: multiple values";
                    properties.payload = 1;
                    {
                        let error = $root.flowcraft.v1.MutationList.verify(message.mutations);
                        if (error)
                            return "mutations." + error;
                    }
                }
                if (message.taskUpdate != null && message.hasOwnProperty("taskUpdate")) {
                    if (properties.payload === 1)
                        return "payload: multiple values";
                    properties.payload = 1;
                    {
                        let error = $root.flowcraft.v1.TaskUpdate.verify(message.taskUpdate);
                        if (error)
                            return "taskUpdate." + error;
                    }
                }
                if (message.streamChunk != null && message.hasOwnProperty("streamChunk")) {
                    if (properties.payload === 1)
                        return "payload: multiple values";
                    properties.payload = 1;
                    {
                        let error = $root.flowcraft.v1.StreamChunk.verify(message.streamChunk);
                        if (error)
                            return "streamChunk." + error;
                    }
                }
                if (message.error != null && message.hasOwnProperty("error")) {
                    if (properties.payload === 1)
                        return "payload: multiple values";
                    properties.payload = 1;
                    {
                        let error = $root.flowcraft.v1.ErrorResponse.verify(message.error);
                        if (error)
                            return "error." + error;
                    }
                }
                return null;
            };

            /**
             * Creates a FlowMessage message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof flowcraft.v1.FlowMessage
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {flowcraft.v1.FlowMessage} FlowMessage
             */
            FlowMessage.fromObject = function fromObject(object) {
                if (object instanceof $root.flowcraft.v1.FlowMessage)
                    return object;
                let message = new $root.flowcraft.v1.FlowMessage();
                if (object.messageId != null)
                    message.messageId = String(object.messageId);
                if (object.timestamp != null)
                    if ($util.Long)
                        (message.timestamp = $util.Long.fromValue(object.timestamp)).unsigned = false;
                    else if (typeof object.timestamp === "string")
                        message.timestamp = parseInt(object.timestamp, 10);
                    else if (typeof object.timestamp === "number")
                        message.timestamp = object.timestamp;
                    else if (typeof object.timestamp === "object")
                        message.timestamp = new $util.LongBits(object.timestamp.low >>> 0, object.timestamp.high >>> 0).toNumber();
                if (object.syncRequest != null) {
                    if (typeof object.syncRequest !== "object")
                        throw TypeError(".flowcraft.v1.FlowMessage.syncRequest: object expected");
                    message.syncRequest = $root.flowcraft.v1.SyncRequest.fromObject(object.syncRequest);
                }
                if (object.yjsUpdate != null)
                    if (typeof object.yjsUpdate === "string")
                        $util.base64.decode(object.yjsUpdate, message.yjsUpdate = $util.newBuffer($util.base64.length(object.yjsUpdate)), 0);
                    else if (object.yjsUpdate.length >= 0)
                        message.yjsUpdate = object.yjsUpdate;
                if (object.nodeUpdate != null) {
                    if (typeof object.nodeUpdate !== "object")
                        throw TypeError(".flowcraft.v1.FlowMessage.nodeUpdate: object expected");
                    message.nodeUpdate = $root.flowcraft.v1.UpdateNodeRequest.fromObject(object.nodeUpdate);
                }
                if (object.widgetUpdate != null) {
                    if (typeof object.widgetUpdate !== "object")
                        throw TypeError(".flowcraft.v1.FlowMessage.widgetUpdate: object expected");
                    message.widgetUpdate = $root.flowcraft.v1.UpdateWidgetRequest.fromObject(object.widgetUpdate);
                }
                if (object.actionExecute != null) {
                    if (typeof object.actionExecute !== "object")
                        throw TypeError(".flowcraft.v1.FlowMessage.actionExecute: object expected");
                    message.actionExecute = $root.flowcraft.v1.ActionExecutionRequest.fromObject(object.actionExecute);
                }
                if (object.taskCancel != null) {
                    if (typeof object.taskCancel !== "object")
                        throw TypeError(".flowcraft.v1.FlowMessage.taskCancel: object expected");
                    message.taskCancel = $root.flowcraft.v1.TaskCancelRequest.fromObject(object.taskCancel);
                }
                if (object.viewportUpdate != null) {
                    if (typeof object.viewportUpdate !== "object")
                        throw TypeError(".flowcraft.v1.FlowMessage.viewportUpdate: object expected");
                    message.viewportUpdate = $root.flowcraft.v1.ViewportUpdate.fromObject(object.viewportUpdate);
                }
                if (object.snapshot != null) {
                    if (typeof object.snapshot !== "object")
                        throw TypeError(".flowcraft.v1.FlowMessage.snapshot: object expected");
                    message.snapshot = $root.flowcraft.v1.GraphSnapshot.fromObject(object.snapshot);
                }
                if (object.mutations != null) {
                    if (typeof object.mutations !== "object")
                        throw TypeError(".flowcraft.v1.FlowMessage.mutations: object expected");
                    message.mutations = $root.flowcraft.v1.MutationList.fromObject(object.mutations);
                }
                if (object.taskUpdate != null) {
                    if (typeof object.taskUpdate !== "object")
                        throw TypeError(".flowcraft.v1.FlowMessage.taskUpdate: object expected");
                    message.taskUpdate = $root.flowcraft.v1.TaskUpdate.fromObject(object.taskUpdate);
                }
                if (object.streamChunk != null) {
                    if (typeof object.streamChunk !== "object")
                        throw TypeError(".flowcraft.v1.FlowMessage.streamChunk: object expected");
                    message.streamChunk = $root.flowcraft.v1.StreamChunk.fromObject(object.streamChunk);
                }
                if (object.error != null) {
                    if (typeof object.error !== "object")
                        throw TypeError(".flowcraft.v1.FlowMessage.error: object expected");
                    message.error = $root.flowcraft.v1.ErrorResponse.fromObject(object.error);
                }
                return message;
            };

            /**
             * Creates a plain object from a FlowMessage message. Also converts values to other types if specified.
             * @function toObject
             * @memberof flowcraft.v1.FlowMessage
             * @static
             * @param {flowcraft.v1.FlowMessage} message FlowMessage
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            FlowMessage.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                let object = {};
                if (options.defaults) {
                    object.messageId = "";
                    if ($util.Long) {
                        let long = new $util.Long(0, 0, false);
                        object.timestamp = options.longs === String ? long.toString() : options.longs === Number ? long.toNumber() : long;
                    } else
                        object.timestamp = options.longs === String ? "0" : 0;
                }
                if (message.messageId != null && message.hasOwnProperty("messageId"))
                    object.messageId = message.messageId;
                if (message.timestamp != null && message.hasOwnProperty("timestamp"))
                    if (typeof message.timestamp === "number")
                        object.timestamp = options.longs === String ? String(message.timestamp) : message.timestamp;
                    else
                        object.timestamp = options.longs === String ? $util.Long.prototype.toString.call(message.timestamp) : options.longs === Number ? new $util.LongBits(message.timestamp.low >>> 0, message.timestamp.high >>> 0).toNumber() : message.timestamp;
                if (message.syncRequest != null && message.hasOwnProperty("syncRequest")) {
                    object.syncRequest = $root.flowcraft.v1.SyncRequest.toObject(message.syncRequest, options);
                    if (options.oneofs)
                        object.payload = "syncRequest";
                }
                if (message.yjsUpdate != null && message.hasOwnProperty("yjsUpdate")) {
                    object.yjsUpdate = options.bytes === String ? $util.base64.encode(message.yjsUpdate, 0, message.yjsUpdate.length) : options.bytes === Array ? Array.prototype.slice.call(message.yjsUpdate) : message.yjsUpdate;
                    if (options.oneofs)
                        object.payload = "yjsUpdate";
                }
                if (message.nodeUpdate != null && message.hasOwnProperty("nodeUpdate")) {
                    object.nodeUpdate = $root.flowcraft.v1.UpdateNodeRequest.toObject(message.nodeUpdate, options);
                    if (options.oneofs)
                        object.payload = "nodeUpdate";
                }
                if (message.widgetUpdate != null && message.hasOwnProperty("widgetUpdate")) {
                    object.widgetUpdate = $root.flowcraft.v1.UpdateWidgetRequest.toObject(message.widgetUpdate, options);
                    if (options.oneofs)
                        object.payload = "widgetUpdate";
                }
                if (message.actionExecute != null && message.hasOwnProperty("actionExecute")) {
                    object.actionExecute = $root.flowcraft.v1.ActionExecutionRequest.toObject(message.actionExecute, options);
                    if (options.oneofs)
                        object.payload = "actionExecute";
                }
                if (message.taskCancel != null && message.hasOwnProperty("taskCancel")) {
                    object.taskCancel = $root.flowcraft.v1.TaskCancelRequest.toObject(message.taskCancel, options);
                    if (options.oneofs)
                        object.payload = "taskCancel";
                }
                if (message.viewportUpdate != null && message.hasOwnProperty("viewportUpdate")) {
                    object.viewportUpdate = $root.flowcraft.v1.ViewportUpdate.toObject(message.viewportUpdate, options);
                    if (options.oneofs)
                        object.payload = "viewportUpdate";
                }
                if (message.snapshot != null && message.hasOwnProperty("snapshot")) {
                    object.snapshot = $root.flowcraft.v1.GraphSnapshot.toObject(message.snapshot, options);
                    if (options.oneofs)
                        object.payload = "snapshot";
                }
                if (message.mutations != null && message.hasOwnProperty("mutations")) {
                    object.mutations = $root.flowcraft.v1.MutationList.toObject(message.mutations, options);
                    if (options.oneofs)
                        object.payload = "mutations";
                }
                if (message.taskUpdate != null && message.hasOwnProperty("taskUpdate")) {
                    object.taskUpdate = $root.flowcraft.v1.TaskUpdate.toObject(message.taskUpdate, options);
                    if (options.oneofs)
                        object.payload = "taskUpdate";
                }
                if (message.streamChunk != null && message.hasOwnProperty("streamChunk")) {
                    object.streamChunk = $root.flowcraft.v1.StreamChunk.toObject(message.streamChunk, options);
                    if (options.oneofs)
                        object.payload = "streamChunk";
                }
                if (message.error != null && message.hasOwnProperty("error")) {
                    object.error = $root.flowcraft.v1.ErrorResponse.toObject(message.error, options);
                    if (options.oneofs)
                        object.payload = "error";
                }
                return object;
            };

            /**
             * Converts this FlowMessage to JSON.
             * @function toJSON
             * @memberof flowcraft.v1.FlowMessage
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            FlowMessage.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };

            /**
             * Gets the default type url for FlowMessage
             * @function getTypeUrl
             * @memberof flowcraft.v1.FlowMessage
             * @static
             * @param {string} [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns {string} The default type url
             */
            FlowMessage.getTypeUrl = function getTypeUrl(typeUrlPrefix) {
                if (typeUrlPrefix === undefined) {
                    typeUrlPrefix = "type.googleapis.com";
                }
                return typeUrlPrefix + "/flowcraft.v1.FlowMessage";
            };

            return FlowMessage;
        })();

        v1.SyncRequest = (function() {

            /**
             * Properties of a SyncRequest.
             * @memberof flowcraft.v1
             * @interface ISyncRequest
             * @property {string|null} [graphId] SyncRequest graphId
             */

            /**
             * Constructs a new SyncRequest.
             * @memberof flowcraft.v1
             * @classdesc Represents a SyncRequest.
             * @implements ISyncRequest
             * @constructor
             * @param {flowcraft.v1.ISyncRequest=} [properties] Properties to set
             */
            function SyncRequest(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * SyncRequest graphId.
             * @member {string} graphId
             * @memberof flowcraft.v1.SyncRequest
             * @instance
             */
            SyncRequest.prototype.graphId = "";

            /**
             * Creates a new SyncRequest instance using the specified properties.
             * @function create
             * @memberof flowcraft.v1.SyncRequest
             * @static
             * @param {flowcraft.v1.ISyncRequest=} [properties] Properties to set
             * @returns {flowcraft.v1.SyncRequest} SyncRequest instance
             */
            SyncRequest.create = function create(properties) {
                return new SyncRequest(properties);
            };

            /**
             * Encodes the specified SyncRequest message. Does not implicitly {@link flowcraft.v1.SyncRequest.verify|verify} messages.
             * @function encode
             * @memberof flowcraft.v1.SyncRequest
             * @static
             * @param {flowcraft.v1.ISyncRequest} message SyncRequest message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            SyncRequest.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.graphId != null && Object.hasOwnProperty.call(message, "graphId"))
                    writer.uint32(/* id 1, wireType 2 =*/10).string(message.graphId);
                return writer;
            };

            /**
             * Encodes the specified SyncRequest message, length delimited. Does not implicitly {@link flowcraft.v1.SyncRequest.verify|verify} messages.
             * @function encodeDelimited
             * @memberof flowcraft.v1.SyncRequest
             * @static
             * @param {flowcraft.v1.ISyncRequest} message SyncRequest message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            SyncRequest.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes a SyncRequest message from the specified reader or buffer.
             * @function decode
             * @memberof flowcraft.v1.SyncRequest
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {flowcraft.v1.SyncRequest} SyncRequest
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            SyncRequest.decode = function decode(reader, length, error) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.flowcraft.v1.SyncRequest();
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    if (tag === error)
                        break;
                    switch (tag >>> 3) {
                    case 1: {
                            message.graphId = reader.string();
                            break;
                        }
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };

            /**
             * Decodes a SyncRequest message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof flowcraft.v1.SyncRequest
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {flowcraft.v1.SyncRequest} SyncRequest
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            SyncRequest.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies a SyncRequest message.
             * @function verify
             * @memberof flowcraft.v1.SyncRequest
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            SyncRequest.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.graphId != null && message.hasOwnProperty("graphId"))
                    if (!$util.isString(message.graphId))
                        return "graphId: string expected";
                return null;
            };

            /**
             * Creates a SyncRequest message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof flowcraft.v1.SyncRequest
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {flowcraft.v1.SyncRequest} SyncRequest
             */
            SyncRequest.fromObject = function fromObject(object) {
                if (object instanceof $root.flowcraft.v1.SyncRequest)
                    return object;
                let message = new $root.flowcraft.v1.SyncRequest();
                if (object.graphId != null)
                    message.graphId = String(object.graphId);
                return message;
            };

            /**
             * Creates a plain object from a SyncRequest message. Also converts values to other types if specified.
             * @function toObject
             * @memberof flowcraft.v1.SyncRequest
             * @static
             * @param {flowcraft.v1.SyncRequest} message SyncRequest
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            SyncRequest.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                let object = {};
                if (options.defaults)
                    object.graphId = "";
                if (message.graphId != null && message.hasOwnProperty("graphId"))
                    object.graphId = message.graphId;
                return object;
            };

            /**
             * Converts this SyncRequest to JSON.
             * @function toJSON
             * @memberof flowcraft.v1.SyncRequest
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            SyncRequest.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };

            /**
             * Gets the default type url for SyncRequest
             * @function getTypeUrl
             * @memberof flowcraft.v1.SyncRequest
             * @static
             * @param {string} [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns {string} The default type url
             */
            SyncRequest.getTypeUrl = function getTypeUrl(typeUrlPrefix) {
                if (typeUrlPrefix === undefined) {
                    typeUrlPrefix = "type.googleapis.com";
                }
                return typeUrlPrefix + "/flowcraft.v1.SyncRequest";
            };

            return SyncRequest;
        })();

        v1.UpdateNodeRequest = (function() {

            /**
             * Properties of an UpdateNodeRequest.
             * @memberof flowcraft.v1
             * @interface IUpdateNodeRequest
             * @property {string|null} [nodeId] UpdateNodeRequest nodeId
             * @property {flowcraft.v1.INodeData|null} [data] UpdateNodeRequest data
             * @property {flowcraft.v1.IPosition|null} [position] UpdateNodeRequest position
             */

            /**
             * Constructs a new UpdateNodeRequest.
             * @memberof flowcraft.v1
             * @classdesc Represents an UpdateNodeRequest.
             * @implements IUpdateNodeRequest
             * @constructor
             * @param {flowcraft.v1.IUpdateNodeRequest=} [properties] Properties to set
             */
            function UpdateNodeRequest(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * UpdateNodeRequest nodeId.
             * @member {string} nodeId
             * @memberof flowcraft.v1.UpdateNodeRequest
             * @instance
             */
            UpdateNodeRequest.prototype.nodeId = "";

            /**
             * UpdateNodeRequest data.
             * @member {flowcraft.v1.INodeData|null|undefined} data
             * @memberof flowcraft.v1.UpdateNodeRequest
             * @instance
             */
            UpdateNodeRequest.prototype.data = null;

            /**
             * UpdateNodeRequest position.
             * @member {flowcraft.v1.IPosition|null|undefined} position
             * @memberof flowcraft.v1.UpdateNodeRequest
             * @instance
             */
            UpdateNodeRequest.prototype.position = null;

            /**
             * Creates a new UpdateNodeRequest instance using the specified properties.
             * @function create
             * @memberof flowcraft.v1.UpdateNodeRequest
             * @static
             * @param {flowcraft.v1.IUpdateNodeRequest=} [properties] Properties to set
             * @returns {flowcraft.v1.UpdateNodeRequest} UpdateNodeRequest instance
             */
            UpdateNodeRequest.create = function create(properties) {
                return new UpdateNodeRequest(properties);
            };

            /**
             * Encodes the specified UpdateNodeRequest message. Does not implicitly {@link flowcraft.v1.UpdateNodeRequest.verify|verify} messages.
             * @function encode
             * @memberof flowcraft.v1.UpdateNodeRequest
             * @static
             * @param {flowcraft.v1.IUpdateNodeRequest} message UpdateNodeRequest message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            UpdateNodeRequest.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.nodeId != null && Object.hasOwnProperty.call(message, "nodeId"))
                    writer.uint32(/* id 1, wireType 2 =*/10).string(message.nodeId);
                if (message.data != null && Object.hasOwnProperty.call(message, "data"))
                    $root.flowcraft.v1.NodeData.encode(message.data, writer.uint32(/* id 2, wireType 2 =*/18).fork()).ldelim();
                if (message.position != null && Object.hasOwnProperty.call(message, "position"))
                    $root.flowcraft.v1.Position.encode(message.position, writer.uint32(/* id 3, wireType 2 =*/26).fork()).ldelim();
                return writer;
            };

            /**
             * Encodes the specified UpdateNodeRequest message, length delimited. Does not implicitly {@link flowcraft.v1.UpdateNodeRequest.verify|verify} messages.
             * @function encodeDelimited
             * @memberof flowcraft.v1.UpdateNodeRequest
             * @static
             * @param {flowcraft.v1.IUpdateNodeRequest} message UpdateNodeRequest message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            UpdateNodeRequest.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes an UpdateNodeRequest message from the specified reader or buffer.
             * @function decode
             * @memberof flowcraft.v1.UpdateNodeRequest
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {flowcraft.v1.UpdateNodeRequest} UpdateNodeRequest
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            UpdateNodeRequest.decode = function decode(reader, length, error) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.flowcraft.v1.UpdateNodeRequest();
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    if (tag === error)
                        break;
                    switch (tag >>> 3) {
                    case 1: {
                            message.nodeId = reader.string();
                            break;
                        }
                    case 2: {
                            message.data = $root.flowcraft.v1.NodeData.decode(reader, reader.uint32());
                            break;
                        }
                    case 3: {
                            message.position = $root.flowcraft.v1.Position.decode(reader, reader.uint32());
                            break;
                        }
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };

            /**
             * Decodes an UpdateNodeRequest message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof flowcraft.v1.UpdateNodeRequest
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {flowcraft.v1.UpdateNodeRequest} UpdateNodeRequest
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            UpdateNodeRequest.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies an UpdateNodeRequest message.
             * @function verify
             * @memberof flowcraft.v1.UpdateNodeRequest
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            UpdateNodeRequest.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.nodeId != null && message.hasOwnProperty("nodeId"))
                    if (!$util.isString(message.nodeId))
                        return "nodeId: string expected";
                if (message.data != null && message.hasOwnProperty("data")) {
                    let error = $root.flowcraft.v1.NodeData.verify(message.data);
                    if (error)
                        return "data." + error;
                }
                if (message.position != null && message.hasOwnProperty("position")) {
                    let error = $root.flowcraft.v1.Position.verify(message.position);
                    if (error)
                        return "position." + error;
                }
                return null;
            };

            /**
             * Creates an UpdateNodeRequest message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof flowcraft.v1.UpdateNodeRequest
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {flowcraft.v1.UpdateNodeRequest} UpdateNodeRequest
             */
            UpdateNodeRequest.fromObject = function fromObject(object) {
                if (object instanceof $root.flowcraft.v1.UpdateNodeRequest)
                    return object;
                let message = new $root.flowcraft.v1.UpdateNodeRequest();
                if (object.nodeId != null)
                    message.nodeId = String(object.nodeId);
                if (object.data != null) {
                    if (typeof object.data !== "object")
                        throw TypeError(".flowcraft.v1.UpdateNodeRequest.data: object expected");
                    message.data = $root.flowcraft.v1.NodeData.fromObject(object.data);
                }
                if (object.position != null) {
                    if (typeof object.position !== "object")
                        throw TypeError(".flowcraft.v1.UpdateNodeRequest.position: object expected");
                    message.position = $root.flowcraft.v1.Position.fromObject(object.position);
                }
                return message;
            };

            /**
             * Creates a plain object from an UpdateNodeRequest message. Also converts values to other types if specified.
             * @function toObject
             * @memberof flowcraft.v1.UpdateNodeRequest
             * @static
             * @param {flowcraft.v1.UpdateNodeRequest} message UpdateNodeRequest
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            UpdateNodeRequest.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                let object = {};
                if (options.defaults) {
                    object.nodeId = "";
                    object.data = null;
                    object.position = null;
                }
                if (message.nodeId != null && message.hasOwnProperty("nodeId"))
                    object.nodeId = message.nodeId;
                if (message.data != null && message.hasOwnProperty("data"))
                    object.data = $root.flowcraft.v1.NodeData.toObject(message.data, options);
                if (message.position != null && message.hasOwnProperty("position"))
                    object.position = $root.flowcraft.v1.Position.toObject(message.position, options);
                return object;
            };

            /**
             * Converts this UpdateNodeRequest to JSON.
             * @function toJSON
             * @memberof flowcraft.v1.UpdateNodeRequest
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            UpdateNodeRequest.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };

            /**
             * Gets the default type url for UpdateNodeRequest
             * @function getTypeUrl
             * @memberof flowcraft.v1.UpdateNodeRequest
             * @static
             * @param {string} [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns {string} The default type url
             */
            UpdateNodeRequest.getTypeUrl = function getTypeUrl(typeUrlPrefix) {
                if (typeUrlPrefix === undefined) {
                    typeUrlPrefix = "type.googleapis.com";
                }
                return typeUrlPrefix + "/flowcraft.v1.UpdateNodeRequest";
            };

            return UpdateNodeRequest;
        })();

        v1.UpdateWidgetRequest = (function() {

            /**
             * Properties of an UpdateWidgetRequest.
             * @memberof flowcraft.v1
             * @interface IUpdateWidgetRequest
             * @property {string|null} [nodeId] UpdateWidgetRequest nodeId
             * @property {string|null} [widgetId] UpdateWidgetRequest widgetId
             * @property {string|null} [valueJson] UpdateWidgetRequest valueJson
             */

            /**
             * Constructs a new UpdateWidgetRequest.
             * @memberof flowcraft.v1
             * @classdesc Represents an UpdateWidgetRequest.
             * @implements IUpdateWidgetRequest
             * @constructor
             * @param {flowcraft.v1.IUpdateWidgetRequest=} [properties] Properties to set
             */
            function UpdateWidgetRequest(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * UpdateWidgetRequest nodeId.
             * @member {string} nodeId
             * @memberof flowcraft.v1.UpdateWidgetRequest
             * @instance
             */
            UpdateWidgetRequest.prototype.nodeId = "";

            /**
             * UpdateWidgetRequest widgetId.
             * @member {string} widgetId
             * @memberof flowcraft.v1.UpdateWidgetRequest
             * @instance
             */
            UpdateWidgetRequest.prototype.widgetId = "";

            /**
             * UpdateWidgetRequest valueJson.
             * @member {string} valueJson
             * @memberof flowcraft.v1.UpdateWidgetRequest
             * @instance
             */
            UpdateWidgetRequest.prototype.valueJson = "";

            /**
             * Creates a new UpdateWidgetRequest instance using the specified properties.
             * @function create
             * @memberof flowcraft.v1.UpdateWidgetRequest
             * @static
             * @param {flowcraft.v1.IUpdateWidgetRequest=} [properties] Properties to set
             * @returns {flowcraft.v1.UpdateWidgetRequest} UpdateWidgetRequest instance
             */
            UpdateWidgetRequest.create = function create(properties) {
                return new UpdateWidgetRequest(properties);
            };

            /**
             * Encodes the specified UpdateWidgetRequest message. Does not implicitly {@link flowcraft.v1.UpdateWidgetRequest.verify|verify} messages.
             * @function encode
             * @memberof flowcraft.v1.UpdateWidgetRequest
             * @static
             * @param {flowcraft.v1.IUpdateWidgetRequest} message UpdateWidgetRequest message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            UpdateWidgetRequest.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.nodeId != null && Object.hasOwnProperty.call(message, "nodeId"))
                    writer.uint32(/* id 1, wireType 2 =*/10).string(message.nodeId);
                if (message.widgetId != null && Object.hasOwnProperty.call(message, "widgetId"))
                    writer.uint32(/* id 2, wireType 2 =*/18).string(message.widgetId);
                if (message.valueJson != null && Object.hasOwnProperty.call(message, "valueJson"))
                    writer.uint32(/* id 3, wireType 2 =*/26).string(message.valueJson);
                return writer;
            };

            /**
             * Encodes the specified UpdateWidgetRequest message, length delimited. Does not implicitly {@link flowcraft.v1.UpdateWidgetRequest.verify|verify} messages.
             * @function encodeDelimited
             * @memberof flowcraft.v1.UpdateWidgetRequest
             * @static
             * @param {flowcraft.v1.IUpdateWidgetRequest} message UpdateWidgetRequest message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            UpdateWidgetRequest.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes an UpdateWidgetRequest message from the specified reader or buffer.
             * @function decode
             * @memberof flowcraft.v1.UpdateWidgetRequest
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {flowcraft.v1.UpdateWidgetRequest} UpdateWidgetRequest
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            UpdateWidgetRequest.decode = function decode(reader, length, error) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.flowcraft.v1.UpdateWidgetRequest();
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    if (tag === error)
                        break;
                    switch (tag >>> 3) {
                    case 1: {
                            message.nodeId = reader.string();
                            break;
                        }
                    case 2: {
                            message.widgetId = reader.string();
                            break;
                        }
                    case 3: {
                            message.valueJson = reader.string();
                            break;
                        }
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };

            /**
             * Decodes an UpdateWidgetRequest message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof flowcraft.v1.UpdateWidgetRequest
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {flowcraft.v1.UpdateWidgetRequest} UpdateWidgetRequest
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            UpdateWidgetRequest.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies an UpdateWidgetRequest message.
             * @function verify
             * @memberof flowcraft.v1.UpdateWidgetRequest
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            UpdateWidgetRequest.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.nodeId != null && message.hasOwnProperty("nodeId"))
                    if (!$util.isString(message.nodeId))
                        return "nodeId: string expected";
                if (message.widgetId != null && message.hasOwnProperty("widgetId"))
                    if (!$util.isString(message.widgetId))
                        return "widgetId: string expected";
                if (message.valueJson != null && message.hasOwnProperty("valueJson"))
                    if (!$util.isString(message.valueJson))
                        return "valueJson: string expected";
                return null;
            };

            /**
             * Creates an UpdateWidgetRequest message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof flowcraft.v1.UpdateWidgetRequest
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {flowcraft.v1.UpdateWidgetRequest} UpdateWidgetRequest
             */
            UpdateWidgetRequest.fromObject = function fromObject(object) {
                if (object instanceof $root.flowcraft.v1.UpdateWidgetRequest)
                    return object;
                let message = new $root.flowcraft.v1.UpdateWidgetRequest();
                if (object.nodeId != null)
                    message.nodeId = String(object.nodeId);
                if (object.widgetId != null)
                    message.widgetId = String(object.widgetId);
                if (object.valueJson != null)
                    message.valueJson = String(object.valueJson);
                return message;
            };

            /**
             * Creates a plain object from an UpdateWidgetRequest message. Also converts values to other types if specified.
             * @function toObject
             * @memberof flowcraft.v1.UpdateWidgetRequest
             * @static
             * @param {flowcraft.v1.UpdateWidgetRequest} message UpdateWidgetRequest
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            UpdateWidgetRequest.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                let object = {};
                if (options.defaults) {
                    object.nodeId = "";
                    object.widgetId = "";
                    object.valueJson = "";
                }
                if (message.nodeId != null && message.hasOwnProperty("nodeId"))
                    object.nodeId = message.nodeId;
                if (message.widgetId != null && message.hasOwnProperty("widgetId"))
                    object.widgetId = message.widgetId;
                if (message.valueJson != null && message.hasOwnProperty("valueJson"))
                    object.valueJson = message.valueJson;
                return object;
            };

            /**
             * Converts this UpdateWidgetRequest to JSON.
             * @function toJSON
             * @memberof flowcraft.v1.UpdateWidgetRequest
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            UpdateWidgetRequest.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };

            /**
             * Gets the default type url for UpdateWidgetRequest
             * @function getTypeUrl
             * @memberof flowcraft.v1.UpdateWidgetRequest
             * @static
             * @param {string} [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns {string} The default type url
             */
            UpdateWidgetRequest.getTypeUrl = function getTypeUrl(typeUrlPrefix) {
                if (typeUrlPrefix === undefined) {
                    typeUrlPrefix = "type.googleapis.com";
                }
                return typeUrlPrefix + "/flowcraft.v1.UpdateWidgetRequest";
            };

            return UpdateWidgetRequest;
        })();

        v1.ViewportUpdate = (function() {

            /**
             * Properties of a ViewportUpdate.
             * @memberof flowcraft.v1
             * @interface IViewportUpdate
             * @property {flowcraft.v1.IViewport|null} [viewport] ViewportUpdate viewport
             * @property {flowcraft.v1.IRect|null} [visibleBounds] ViewportUpdate visibleBounds
             */

            /**
             * Constructs a new ViewportUpdate.
             * @memberof flowcraft.v1
             * @classdesc Represents a ViewportUpdate.
             * @implements IViewportUpdate
             * @constructor
             * @param {flowcraft.v1.IViewportUpdate=} [properties] Properties to set
             */
            function ViewportUpdate(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * ViewportUpdate viewport.
             * @member {flowcraft.v1.IViewport|null|undefined} viewport
             * @memberof flowcraft.v1.ViewportUpdate
             * @instance
             */
            ViewportUpdate.prototype.viewport = null;

            /**
             * ViewportUpdate visibleBounds.
             * @member {flowcraft.v1.IRect|null|undefined} visibleBounds
             * @memberof flowcraft.v1.ViewportUpdate
             * @instance
             */
            ViewportUpdate.prototype.visibleBounds = null;

            /**
             * Creates a new ViewportUpdate instance using the specified properties.
             * @function create
             * @memberof flowcraft.v1.ViewportUpdate
             * @static
             * @param {flowcraft.v1.IViewportUpdate=} [properties] Properties to set
             * @returns {flowcraft.v1.ViewportUpdate} ViewportUpdate instance
             */
            ViewportUpdate.create = function create(properties) {
                return new ViewportUpdate(properties);
            };

            /**
             * Encodes the specified ViewportUpdate message. Does not implicitly {@link flowcraft.v1.ViewportUpdate.verify|verify} messages.
             * @function encode
             * @memberof flowcraft.v1.ViewportUpdate
             * @static
             * @param {flowcraft.v1.IViewportUpdate} message ViewportUpdate message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            ViewportUpdate.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.viewport != null && Object.hasOwnProperty.call(message, "viewport"))
                    $root.flowcraft.v1.Viewport.encode(message.viewport, writer.uint32(/* id 1, wireType 2 =*/10).fork()).ldelim();
                if (message.visibleBounds != null && Object.hasOwnProperty.call(message, "visibleBounds"))
                    $root.flowcraft.v1.Rect.encode(message.visibleBounds, writer.uint32(/* id 2, wireType 2 =*/18).fork()).ldelim();
                return writer;
            };

            /**
             * Encodes the specified ViewportUpdate message, length delimited. Does not implicitly {@link flowcraft.v1.ViewportUpdate.verify|verify} messages.
             * @function encodeDelimited
             * @memberof flowcraft.v1.ViewportUpdate
             * @static
             * @param {flowcraft.v1.IViewportUpdate} message ViewportUpdate message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            ViewportUpdate.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes a ViewportUpdate message from the specified reader or buffer.
             * @function decode
             * @memberof flowcraft.v1.ViewportUpdate
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {flowcraft.v1.ViewportUpdate} ViewportUpdate
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            ViewportUpdate.decode = function decode(reader, length, error) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.flowcraft.v1.ViewportUpdate();
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    if (tag === error)
                        break;
                    switch (tag >>> 3) {
                    case 1: {
                            message.viewport = $root.flowcraft.v1.Viewport.decode(reader, reader.uint32());
                            break;
                        }
                    case 2: {
                            message.visibleBounds = $root.flowcraft.v1.Rect.decode(reader, reader.uint32());
                            break;
                        }
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };

            /**
             * Decodes a ViewportUpdate message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof flowcraft.v1.ViewportUpdate
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {flowcraft.v1.ViewportUpdate} ViewportUpdate
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            ViewportUpdate.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies a ViewportUpdate message.
             * @function verify
             * @memberof flowcraft.v1.ViewportUpdate
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            ViewportUpdate.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.viewport != null && message.hasOwnProperty("viewport")) {
                    let error = $root.flowcraft.v1.Viewport.verify(message.viewport);
                    if (error)
                        return "viewport." + error;
                }
                if (message.visibleBounds != null && message.hasOwnProperty("visibleBounds")) {
                    let error = $root.flowcraft.v1.Rect.verify(message.visibleBounds);
                    if (error)
                        return "visibleBounds." + error;
                }
                return null;
            };

            /**
             * Creates a ViewportUpdate message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof flowcraft.v1.ViewportUpdate
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {flowcraft.v1.ViewportUpdate} ViewportUpdate
             */
            ViewportUpdate.fromObject = function fromObject(object) {
                if (object instanceof $root.flowcraft.v1.ViewportUpdate)
                    return object;
                let message = new $root.flowcraft.v1.ViewportUpdate();
                if (object.viewport != null) {
                    if (typeof object.viewport !== "object")
                        throw TypeError(".flowcraft.v1.ViewportUpdate.viewport: object expected");
                    message.viewport = $root.flowcraft.v1.Viewport.fromObject(object.viewport);
                }
                if (object.visibleBounds != null) {
                    if (typeof object.visibleBounds !== "object")
                        throw TypeError(".flowcraft.v1.ViewportUpdate.visibleBounds: object expected");
                    message.visibleBounds = $root.flowcraft.v1.Rect.fromObject(object.visibleBounds);
                }
                return message;
            };

            /**
             * Creates a plain object from a ViewportUpdate message. Also converts values to other types if specified.
             * @function toObject
             * @memberof flowcraft.v1.ViewportUpdate
             * @static
             * @param {flowcraft.v1.ViewportUpdate} message ViewportUpdate
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            ViewportUpdate.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                let object = {};
                if (options.defaults) {
                    object.viewport = null;
                    object.visibleBounds = null;
                }
                if (message.viewport != null && message.hasOwnProperty("viewport"))
                    object.viewport = $root.flowcraft.v1.Viewport.toObject(message.viewport, options);
                if (message.visibleBounds != null && message.hasOwnProperty("visibleBounds"))
                    object.visibleBounds = $root.flowcraft.v1.Rect.toObject(message.visibleBounds, options);
                return object;
            };

            /**
             * Converts this ViewportUpdate to JSON.
             * @function toJSON
             * @memberof flowcraft.v1.ViewportUpdate
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            ViewportUpdate.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };

            /**
             * Gets the default type url for ViewportUpdate
             * @function getTypeUrl
             * @memberof flowcraft.v1.ViewportUpdate
             * @static
             * @param {string} [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns {string} The default type url
             */
            ViewportUpdate.getTypeUrl = function getTypeUrl(typeUrlPrefix) {
                if (typeUrlPrefix === undefined) {
                    typeUrlPrefix = "type.googleapis.com";
                }
                return typeUrlPrefix + "/flowcraft.v1.ViewportUpdate";
            };

            return ViewportUpdate;
        })();

        v1.MutationList = (function() {

            /**
             * Properties of a MutationList.
             * @memberof flowcraft.v1
             * @interface IMutationList
             * @property {Array.<flowcraft.v1.IGraphMutation>|null} [mutations] MutationList mutations
             * @property {number|Long|null} [sequenceNumber] MutationList sequenceNumber
             */

            /**
             * Constructs a new MutationList.
             * @memberof flowcraft.v1
             * @classdesc Represents a MutationList.
             * @implements IMutationList
             * @constructor
             * @param {flowcraft.v1.IMutationList=} [properties] Properties to set
             */
            function MutationList(properties) {
                this.mutations = [];
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * MutationList mutations.
             * @member {Array.<flowcraft.v1.IGraphMutation>} mutations
             * @memberof flowcraft.v1.MutationList
             * @instance
             */
            MutationList.prototype.mutations = $util.emptyArray;

            /**
             * MutationList sequenceNumber.
             * @member {number|Long} sequenceNumber
             * @memberof flowcraft.v1.MutationList
             * @instance
             */
            MutationList.prototype.sequenceNumber = $util.Long ? $util.Long.fromBits(0,0,false) : 0;

            /**
             * Creates a new MutationList instance using the specified properties.
             * @function create
             * @memberof flowcraft.v1.MutationList
             * @static
             * @param {flowcraft.v1.IMutationList=} [properties] Properties to set
             * @returns {flowcraft.v1.MutationList} MutationList instance
             */
            MutationList.create = function create(properties) {
                return new MutationList(properties);
            };

            /**
             * Encodes the specified MutationList message. Does not implicitly {@link flowcraft.v1.MutationList.verify|verify} messages.
             * @function encode
             * @memberof flowcraft.v1.MutationList
             * @static
             * @param {flowcraft.v1.IMutationList} message MutationList message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            MutationList.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.mutations != null && message.mutations.length)
                    for (let i = 0; i < message.mutations.length; ++i)
                        $root.flowcraft.v1.GraphMutation.encode(message.mutations[i], writer.uint32(/* id 1, wireType 2 =*/10).fork()).ldelim();
                if (message.sequenceNumber != null && Object.hasOwnProperty.call(message, "sequenceNumber"))
                    writer.uint32(/* id 2, wireType 0 =*/16).int64(message.sequenceNumber);
                return writer;
            };

            /**
             * Encodes the specified MutationList message, length delimited. Does not implicitly {@link flowcraft.v1.MutationList.verify|verify} messages.
             * @function encodeDelimited
             * @memberof flowcraft.v1.MutationList
             * @static
             * @param {flowcraft.v1.IMutationList} message MutationList message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            MutationList.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes a MutationList message from the specified reader or buffer.
             * @function decode
             * @memberof flowcraft.v1.MutationList
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {flowcraft.v1.MutationList} MutationList
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            MutationList.decode = function decode(reader, length, error) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.flowcraft.v1.MutationList();
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    if (tag === error)
                        break;
                    switch (tag >>> 3) {
                    case 1: {
                            if (!(message.mutations && message.mutations.length))
                                message.mutations = [];
                            message.mutations.push($root.flowcraft.v1.GraphMutation.decode(reader, reader.uint32()));
                            break;
                        }
                    case 2: {
                            message.sequenceNumber = reader.int64();
                            break;
                        }
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };

            /**
             * Decodes a MutationList message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof flowcraft.v1.MutationList
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {flowcraft.v1.MutationList} MutationList
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            MutationList.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies a MutationList message.
             * @function verify
             * @memberof flowcraft.v1.MutationList
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            MutationList.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.mutations != null && message.hasOwnProperty("mutations")) {
                    if (!Array.isArray(message.mutations))
                        return "mutations: array expected";
                    for (let i = 0; i < message.mutations.length; ++i) {
                        let error = $root.flowcraft.v1.GraphMutation.verify(message.mutations[i]);
                        if (error)
                            return "mutations." + error;
                    }
                }
                if (message.sequenceNumber != null && message.hasOwnProperty("sequenceNumber"))
                    if (!$util.isInteger(message.sequenceNumber) && !(message.sequenceNumber && $util.isInteger(message.sequenceNumber.low) && $util.isInteger(message.sequenceNumber.high)))
                        return "sequenceNumber: integer|Long expected";
                return null;
            };

            /**
             * Creates a MutationList message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof flowcraft.v1.MutationList
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {flowcraft.v1.MutationList} MutationList
             */
            MutationList.fromObject = function fromObject(object) {
                if (object instanceof $root.flowcraft.v1.MutationList)
                    return object;
                let message = new $root.flowcraft.v1.MutationList();
                if (object.mutations) {
                    if (!Array.isArray(object.mutations))
                        throw TypeError(".flowcraft.v1.MutationList.mutations: array expected");
                    message.mutations = [];
                    for (let i = 0; i < object.mutations.length; ++i) {
                        if (typeof object.mutations[i] !== "object")
                            throw TypeError(".flowcraft.v1.MutationList.mutations: object expected");
                        message.mutations[i] = $root.flowcraft.v1.GraphMutation.fromObject(object.mutations[i]);
                    }
                }
                if (object.sequenceNumber != null)
                    if ($util.Long)
                        (message.sequenceNumber = $util.Long.fromValue(object.sequenceNumber)).unsigned = false;
                    else if (typeof object.sequenceNumber === "string")
                        message.sequenceNumber = parseInt(object.sequenceNumber, 10);
                    else if (typeof object.sequenceNumber === "number")
                        message.sequenceNumber = object.sequenceNumber;
                    else if (typeof object.sequenceNumber === "object")
                        message.sequenceNumber = new $util.LongBits(object.sequenceNumber.low >>> 0, object.sequenceNumber.high >>> 0).toNumber();
                return message;
            };

            /**
             * Creates a plain object from a MutationList message. Also converts values to other types if specified.
             * @function toObject
             * @memberof flowcraft.v1.MutationList
             * @static
             * @param {flowcraft.v1.MutationList} message MutationList
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            MutationList.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                let object = {};
                if (options.arrays || options.defaults)
                    object.mutations = [];
                if (options.defaults)
                    if ($util.Long) {
                        let long = new $util.Long(0, 0, false);
                        object.sequenceNumber = options.longs === String ? long.toString() : options.longs === Number ? long.toNumber() : long;
                    } else
                        object.sequenceNumber = options.longs === String ? "0" : 0;
                if (message.mutations && message.mutations.length) {
                    object.mutations = [];
                    for (let j = 0; j < message.mutations.length; ++j)
                        object.mutations[j] = $root.flowcraft.v1.GraphMutation.toObject(message.mutations[j], options);
                }
                if (message.sequenceNumber != null && message.hasOwnProperty("sequenceNumber"))
                    if (typeof message.sequenceNumber === "number")
                        object.sequenceNumber = options.longs === String ? String(message.sequenceNumber) : message.sequenceNumber;
                    else
                        object.sequenceNumber = options.longs === String ? $util.Long.prototype.toString.call(message.sequenceNumber) : options.longs === Number ? new $util.LongBits(message.sequenceNumber.low >>> 0, message.sequenceNumber.high >>> 0).toNumber() : message.sequenceNumber;
                return object;
            };

            /**
             * Converts this MutationList to JSON.
             * @function toJSON
             * @memberof flowcraft.v1.MutationList
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            MutationList.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };

            /**
             * Gets the default type url for MutationList
             * @function getTypeUrl
             * @memberof flowcraft.v1.MutationList
             * @static
             * @param {string} [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns {string} The default type url
             */
            MutationList.getTypeUrl = function getTypeUrl(typeUrlPrefix) {
                if (typeUrlPrefix === undefined) {
                    typeUrlPrefix = "type.googleapis.com";
                }
                return typeUrlPrefix + "/flowcraft.v1.MutationList";
            };

            return MutationList;
        })();

        v1.StreamChunk = (function() {

            /**
             * Properties of a StreamChunk.
             * @memberof flowcraft.v1
             * @interface IStreamChunk
             * @property {string|null} [nodeId] StreamChunk nodeId
             * @property {string|null} [widgetId] StreamChunk widgetId
             * @property {string|null} [chunkData] StreamChunk chunkData
             * @property {boolean|null} [isDone] StreamChunk isDone
             */

            /**
             * Constructs a new StreamChunk.
             * @memberof flowcraft.v1
             * @classdesc Represents a StreamChunk.
             * @implements IStreamChunk
             * @constructor
             * @param {flowcraft.v1.IStreamChunk=} [properties] Properties to set
             */
            function StreamChunk(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * StreamChunk nodeId.
             * @member {string} nodeId
             * @memberof flowcraft.v1.StreamChunk
             * @instance
             */
            StreamChunk.prototype.nodeId = "";

            /**
             * StreamChunk widgetId.
             * @member {string} widgetId
             * @memberof flowcraft.v1.StreamChunk
             * @instance
             */
            StreamChunk.prototype.widgetId = "";

            /**
             * StreamChunk chunkData.
             * @member {string} chunkData
             * @memberof flowcraft.v1.StreamChunk
             * @instance
             */
            StreamChunk.prototype.chunkData = "";

            /**
             * StreamChunk isDone.
             * @member {boolean} isDone
             * @memberof flowcraft.v1.StreamChunk
             * @instance
             */
            StreamChunk.prototype.isDone = false;

            /**
             * Creates a new StreamChunk instance using the specified properties.
             * @function create
             * @memberof flowcraft.v1.StreamChunk
             * @static
             * @param {flowcraft.v1.IStreamChunk=} [properties] Properties to set
             * @returns {flowcraft.v1.StreamChunk} StreamChunk instance
             */
            StreamChunk.create = function create(properties) {
                return new StreamChunk(properties);
            };

            /**
             * Encodes the specified StreamChunk message. Does not implicitly {@link flowcraft.v1.StreamChunk.verify|verify} messages.
             * @function encode
             * @memberof flowcraft.v1.StreamChunk
             * @static
             * @param {flowcraft.v1.IStreamChunk} message StreamChunk message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            StreamChunk.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.nodeId != null && Object.hasOwnProperty.call(message, "nodeId"))
                    writer.uint32(/* id 1, wireType 2 =*/10).string(message.nodeId);
                if (message.widgetId != null && Object.hasOwnProperty.call(message, "widgetId"))
                    writer.uint32(/* id 2, wireType 2 =*/18).string(message.widgetId);
                if (message.chunkData != null && Object.hasOwnProperty.call(message, "chunkData"))
                    writer.uint32(/* id 3, wireType 2 =*/26).string(message.chunkData);
                if (message.isDone != null && Object.hasOwnProperty.call(message, "isDone"))
                    writer.uint32(/* id 4, wireType 0 =*/32).bool(message.isDone);
                return writer;
            };

            /**
             * Encodes the specified StreamChunk message, length delimited. Does not implicitly {@link flowcraft.v1.StreamChunk.verify|verify} messages.
             * @function encodeDelimited
             * @memberof flowcraft.v1.StreamChunk
             * @static
             * @param {flowcraft.v1.IStreamChunk} message StreamChunk message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            StreamChunk.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes a StreamChunk message from the specified reader or buffer.
             * @function decode
             * @memberof flowcraft.v1.StreamChunk
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {flowcraft.v1.StreamChunk} StreamChunk
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            StreamChunk.decode = function decode(reader, length, error) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.flowcraft.v1.StreamChunk();
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    if (tag === error)
                        break;
                    switch (tag >>> 3) {
                    case 1: {
                            message.nodeId = reader.string();
                            break;
                        }
                    case 2: {
                            message.widgetId = reader.string();
                            break;
                        }
                    case 3: {
                            message.chunkData = reader.string();
                            break;
                        }
                    case 4: {
                            message.isDone = reader.bool();
                            break;
                        }
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };

            /**
             * Decodes a StreamChunk message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof flowcraft.v1.StreamChunk
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {flowcraft.v1.StreamChunk} StreamChunk
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            StreamChunk.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies a StreamChunk message.
             * @function verify
             * @memberof flowcraft.v1.StreamChunk
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            StreamChunk.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.nodeId != null && message.hasOwnProperty("nodeId"))
                    if (!$util.isString(message.nodeId))
                        return "nodeId: string expected";
                if (message.widgetId != null && message.hasOwnProperty("widgetId"))
                    if (!$util.isString(message.widgetId))
                        return "widgetId: string expected";
                if (message.chunkData != null && message.hasOwnProperty("chunkData"))
                    if (!$util.isString(message.chunkData))
                        return "chunkData: string expected";
                if (message.isDone != null && message.hasOwnProperty("isDone"))
                    if (typeof message.isDone !== "boolean")
                        return "isDone: boolean expected";
                return null;
            };

            /**
             * Creates a StreamChunk message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof flowcraft.v1.StreamChunk
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {flowcraft.v1.StreamChunk} StreamChunk
             */
            StreamChunk.fromObject = function fromObject(object) {
                if (object instanceof $root.flowcraft.v1.StreamChunk)
                    return object;
                let message = new $root.flowcraft.v1.StreamChunk();
                if (object.nodeId != null)
                    message.nodeId = String(object.nodeId);
                if (object.widgetId != null)
                    message.widgetId = String(object.widgetId);
                if (object.chunkData != null)
                    message.chunkData = String(object.chunkData);
                if (object.isDone != null)
                    message.isDone = Boolean(object.isDone);
                return message;
            };

            /**
             * Creates a plain object from a StreamChunk message. Also converts values to other types if specified.
             * @function toObject
             * @memberof flowcraft.v1.StreamChunk
             * @static
             * @param {flowcraft.v1.StreamChunk} message StreamChunk
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            StreamChunk.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                let object = {};
                if (options.defaults) {
                    object.nodeId = "";
                    object.widgetId = "";
                    object.chunkData = "";
                    object.isDone = false;
                }
                if (message.nodeId != null && message.hasOwnProperty("nodeId"))
                    object.nodeId = message.nodeId;
                if (message.widgetId != null && message.hasOwnProperty("widgetId"))
                    object.widgetId = message.widgetId;
                if (message.chunkData != null && message.hasOwnProperty("chunkData"))
                    object.chunkData = message.chunkData;
                if (message.isDone != null && message.hasOwnProperty("isDone"))
                    object.isDone = message.isDone;
                return object;
            };

            /**
             * Converts this StreamChunk to JSON.
             * @function toJSON
             * @memberof flowcraft.v1.StreamChunk
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            StreamChunk.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };

            /**
             * Gets the default type url for StreamChunk
             * @function getTypeUrl
             * @memberof flowcraft.v1.StreamChunk
             * @static
             * @param {string} [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns {string} The default type url
             */
            StreamChunk.getTypeUrl = function getTypeUrl(typeUrlPrefix) {
                if (typeUrlPrefix === undefined) {
                    typeUrlPrefix = "type.googleapis.com";
                }
                return typeUrlPrefix + "/flowcraft.v1.StreamChunk";
            };

            return StreamChunk;
        })();

        v1.ErrorResponse = (function() {

            /**
             * Properties of an ErrorResponse.
             * @memberof flowcraft.v1
             * @interface IErrorResponse
             * @property {string|null} [code] ErrorResponse code
             * @property {string|null} [message] ErrorResponse message
             */

            /**
             * Constructs a new ErrorResponse.
             * @memberof flowcraft.v1
             * @classdesc Represents an ErrorResponse.
             * @implements IErrorResponse
             * @constructor
             * @param {flowcraft.v1.IErrorResponse=} [properties] Properties to set
             */
            function ErrorResponse(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * ErrorResponse code.
             * @member {string} code
             * @memberof flowcraft.v1.ErrorResponse
             * @instance
             */
            ErrorResponse.prototype.code = "";

            /**
             * ErrorResponse message.
             * @member {string} message
             * @memberof flowcraft.v1.ErrorResponse
             * @instance
             */
            ErrorResponse.prototype.message = "";

            /**
             * Creates a new ErrorResponse instance using the specified properties.
             * @function create
             * @memberof flowcraft.v1.ErrorResponse
             * @static
             * @param {flowcraft.v1.IErrorResponse=} [properties] Properties to set
             * @returns {flowcraft.v1.ErrorResponse} ErrorResponse instance
             */
            ErrorResponse.create = function create(properties) {
                return new ErrorResponse(properties);
            };

            /**
             * Encodes the specified ErrorResponse message. Does not implicitly {@link flowcraft.v1.ErrorResponse.verify|verify} messages.
             * @function encode
             * @memberof flowcraft.v1.ErrorResponse
             * @static
             * @param {flowcraft.v1.IErrorResponse} message ErrorResponse message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            ErrorResponse.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.code != null && Object.hasOwnProperty.call(message, "code"))
                    writer.uint32(/* id 1, wireType 2 =*/10).string(message.code);
                if (message.message != null && Object.hasOwnProperty.call(message, "message"))
                    writer.uint32(/* id 2, wireType 2 =*/18).string(message.message);
                return writer;
            };

            /**
             * Encodes the specified ErrorResponse message, length delimited. Does not implicitly {@link flowcraft.v1.ErrorResponse.verify|verify} messages.
             * @function encodeDelimited
             * @memberof flowcraft.v1.ErrorResponse
             * @static
             * @param {flowcraft.v1.IErrorResponse} message ErrorResponse message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            ErrorResponse.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes an ErrorResponse message from the specified reader or buffer.
             * @function decode
             * @memberof flowcraft.v1.ErrorResponse
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {flowcraft.v1.ErrorResponse} ErrorResponse
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            ErrorResponse.decode = function decode(reader, length, error) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.flowcraft.v1.ErrorResponse();
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    if (tag === error)
                        break;
                    switch (tag >>> 3) {
                    case 1: {
                            message.code = reader.string();
                            break;
                        }
                    case 2: {
                            message.message = reader.string();
                            break;
                        }
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };

            /**
             * Decodes an ErrorResponse message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof flowcraft.v1.ErrorResponse
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {flowcraft.v1.ErrorResponse} ErrorResponse
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            ErrorResponse.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies an ErrorResponse message.
             * @function verify
             * @memberof flowcraft.v1.ErrorResponse
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            ErrorResponse.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.code != null && message.hasOwnProperty("code"))
                    if (!$util.isString(message.code))
                        return "code: string expected";
                if (message.message != null && message.hasOwnProperty("message"))
                    if (!$util.isString(message.message))
                        return "message: string expected";
                return null;
            };

            /**
             * Creates an ErrorResponse message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof flowcraft.v1.ErrorResponse
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {flowcraft.v1.ErrorResponse} ErrorResponse
             */
            ErrorResponse.fromObject = function fromObject(object) {
                if (object instanceof $root.flowcraft.v1.ErrorResponse)
                    return object;
                let message = new $root.flowcraft.v1.ErrorResponse();
                if (object.code != null)
                    message.code = String(object.code);
                if (object.message != null)
                    message.message = String(object.message);
                return message;
            };

            /**
             * Creates a plain object from an ErrorResponse message. Also converts values to other types if specified.
             * @function toObject
             * @memberof flowcraft.v1.ErrorResponse
             * @static
             * @param {flowcraft.v1.ErrorResponse} message ErrorResponse
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            ErrorResponse.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                let object = {};
                if (options.defaults) {
                    object.code = "";
                    object.message = "";
                }
                if (message.code != null && message.hasOwnProperty("code"))
                    object.code = message.code;
                if (message.message != null && message.hasOwnProperty("message"))
                    object.message = message.message;
                return object;
            };

            /**
             * Converts this ErrorResponse to JSON.
             * @function toJSON
             * @memberof flowcraft.v1.ErrorResponse
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            ErrorResponse.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };

            /**
             * Gets the default type url for ErrorResponse
             * @function getTypeUrl
             * @memberof flowcraft.v1.ErrorResponse
             * @static
             * @param {string} [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns {string} The default type url
             */
            ErrorResponse.getTypeUrl = function getTypeUrl(typeUrlPrefix) {
                if (typeUrlPrefix === undefined) {
                    typeUrlPrefix = "type.googleapis.com";
                }
                return typeUrlPrefix + "/flowcraft.v1.ErrorResponse";
            };

            return ErrorResponse;
        })();

        v1.FlowService = (function() {

            /**
             * Constructs a new FlowService service.
             * @memberof flowcraft.v1
             * @classdesc Represents a FlowService
             * @extends $protobuf.rpc.Service
             * @constructor
             * @param {$protobuf.RPCImpl} rpcImpl RPC implementation
             * @param {boolean} [requestDelimited=false] Whether requests are length-delimited
             * @param {boolean} [responseDelimited=false] Whether responses are length-delimited
             */
            function FlowService(rpcImpl, requestDelimited, responseDelimited) {
                $protobuf.rpc.Service.call(this, rpcImpl, requestDelimited, responseDelimited);
            }

            (FlowService.prototype = Object.create($protobuf.rpc.Service.prototype)).constructor = FlowService;

            /**
             * Creates new FlowService service using the specified rpc implementation.
             * @function create
             * @memberof flowcraft.v1.FlowService
             * @static
             * @param {$protobuf.RPCImpl} rpcImpl RPC implementation
             * @param {boolean} [requestDelimited=false] Whether requests are length-delimited
             * @param {boolean} [responseDelimited=false] Whether responses are length-delimited
             * @returns {FlowService} RPC service. Useful where requests and/or responses are streamed.
             */
            FlowService.create = function create(rpcImpl, requestDelimited, responseDelimited) {
                return new this(rpcImpl, requestDelimited, responseDelimited);
            };

            /**
             * Callback as used by {@link flowcraft.v1.FlowService#connect}.
             * @memberof flowcraft.v1.FlowService
             * @typedef ConnectCallback
             * @type {function}
             * @param {Error|null} error Error, if any
             * @param {flowcraft.v1.FlowMessage} [response] FlowMessage
             */

            /**
             * Calls Connect.
             * @function connect
             * @memberof flowcraft.v1.FlowService
             * @instance
             * @param {flowcraft.v1.IFlowMessage} request FlowMessage message or plain object
             * @param {flowcraft.v1.FlowService.ConnectCallback} callback Node-style callback called with the error, if any, and FlowMessage
             * @returns {undefined}
             * @variation 1
             */
            Object.defineProperty(FlowService.prototype.connect = function connect(request, callback) {
                return this.rpcCall(connect, $root.flowcraft.v1.FlowMessage, $root.flowcraft.v1.FlowMessage, request, callback);
            }, "name", { value: "Connect" });

            /**
             * Calls Connect.
             * @function connect
             * @memberof flowcraft.v1.FlowService
             * @instance
             * @param {flowcraft.v1.IFlowMessage} request FlowMessage message or plain object
             * @returns {Promise<flowcraft.v1.FlowMessage>} Promise
             * @variation 2
             */

            return FlowService;
        })();

        v1.GraphSnapshot = (function() {

            /**
             * Properties of a GraphSnapshot.
             * @memberof flowcraft.v1
             * @interface IGraphSnapshot
             * @property {Array.<flowcraft.v1.INode>|null} [nodes] GraphSnapshot nodes
             * @property {Array.<flowcraft.v1.IEdge>|null} [edges] GraphSnapshot edges
             * @property {number|Long|null} [version] GraphSnapshot version
             */

            /**
             * Constructs a new GraphSnapshot.
             * @memberof flowcraft.v1
             * @classdesc Represents a GraphSnapshot.
             * @implements IGraphSnapshot
             * @constructor
             * @param {flowcraft.v1.IGraphSnapshot=} [properties] Properties to set
             */
            function GraphSnapshot(properties) {
                this.nodes = [];
                this.edges = [];
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * GraphSnapshot nodes.
             * @member {Array.<flowcraft.v1.INode>} nodes
             * @memberof flowcraft.v1.GraphSnapshot
             * @instance
             */
            GraphSnapshot.prototype.nodes = $util.emptyArray;

            /**
             * GraphSnapshot edges.
             * @member {Array.<flowcraft.v1.IEdge>} edges
             * @memberof flowcraft.v1.GraphSnapshot
             * @instance
             */
            GraphSnapshot.prototype.edges = $util.emptyArray;

            /**
             * GraphSnapshot version.
             * @member {number|Long} version
             * @memberof flowcraft.v1.GraphSnapshot
             * @instance
             */
            GraphSnapshot.prototype.version = $util.Long ? $util.Long.fromBits(0,0,false) : 0;

            /**
             * Creates a new GraphSnapshot instance using the specified properties.
             * @function create
             * @memberof flowcraft.v1.GraphSnapshot
             * @static
             * @param {flowcraft.v1.IGraphSnapshot=} [properties] Properties to set
             * @returns {flowcraft.v1.GraphSnapshot} GraphSnapshot instance
             */
            GraphSnapshot.create = function create(properties) {
                return new GraphSnapshot(properties);
            };

            /**
             * Encodes the specified GraphSnapshot message. Does not implicitly {@link flowcraft.v1.GraphSnapshot.verify|verify} messages.
             * @function encode
             * @memberof flowcraft.v1.GraphSnapshot
             * @static
             * @param {flowcraft.v1.IGraphSnapshot} message GraphSnapshot message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            GraphSnapshot.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.nodes != null && message.nodes.length)
                    for (let i = 0; i < message.nodes.length; ++i)
                        $root.flowcraft.v1.Node.encode(message.nodes[i], writer.uint32(/* id 1, wireType 2 =*/10).fork()).ldelim();
                if (message.edges != null && message.edges.length)
                    for (let i = 0; i < message.edges.length; ++i)
                        $root.flowcraft.v1.Edge.encode(message.edges[i], writer.uint32(/* id 2, wireType 2 =*/18).fork()).ldelim();
                if (message.version != null && Object.hasOwnProperty.call(message, "version"))
                    writer.uint32(/* id 3, wireType 0 =*/24).int64(message.version);
                return writer;
            };

            /**
             * Encodes the specified GraphSnapshot message, length delimited. Does not implicitly {@link flowcraft.v1.GraphSnapshot.verify|verify} messages.
             * @function encodeDelimited
             * @memberof flowcraft.v1.GraphSnapshot
             * @static
             * @param {flowcraft.v1.IGraphSnapshot} message GraphSnapshot message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            GraphSnapshot.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes a GraphSnapshot message from the specified reader or buffer.
             * @function decode
             * @memberof flowcraft.v1.GraphSnapshot
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {flowcraft.v1.GraphSnapshot} GraphSnapshot
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            GraphSnapshot.decode = function decode(reader, length, error) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.flowcraft.v1.GraphSnapshot();
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    if (tag === error)
                        break;
                    switch (tag >>> 3) {
                    case 1: {
                            if (!(message.nodes && message.nodes.length))
                                message.nodes = [];
                            message.nodes.push($root.flowcraft.v1.Node.decode(reader, reader.uint32()));
                            break;
                        }
                    case 2: {
                            if (!(message.edges && message.edges.length))
                                message.edges = [];
                            message.edges.push($root.flowcraft.v1.Edge.decode(reader, reader.uint32()));
                            break;
                        }
                    case 3: {
                            message.version = reader.int64();
                            break;
                        }
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };

            /**
             * Decodes a GraphSnapshot message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof flowcraft.v1.GraphSnapshot
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {flowcraft.v1.GraphSnapshot} GraphSnapshot
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            GraphSnapshot.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies a GraphSnapshot message.
             * @function verify
             * @memberof flowcraft.v1.GraphSnapshot
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            GraphSnapshot.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.nodes != null && message.hasOwnProperty("nodes")) {
                    if (!Array.isArray(message.nodes))
                        return "nodes: array expected";
                    for (let i = 0; i < message.nodes.length; ++i) {
                        let error = $root.flowcraft.v1.Node.verify(message.nodes[i]);
                        if (error)
                            return "nodes." + error;
                    }
                }
                if (message.edges != null && message.hasOwnProperty("edges")) {
                    if (!Array.isArray(message.edges))
                        return "edges: array expected";
                    for (let i = 0; i < message.edges.length; ++i) {
                        let error = $root.flowcraft.v1.Edge.verify(message.edges[i]);
                        if (error)
                            return "edges." + error;
                    }
                }
                if (message.version != null && message.hasOwnProperty("version"))
                    if (!$util.isInteger(message.version) && !(message.version && $util.isInteger(message.version.low) && $util.isInteger(message.version.high)))
                        return "version: integer|Long expected";
                return null;
            };

            /**
             * Creates a GraphSnapshot message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof flowcraft.v1.GraphSnapshot
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {flowcraft.v1.GraphSnapshot} GraphSnapshot
             */
            GraphSnapshot.fromObject = function fromObject(object) {
                if (object instanceof $root.flowcraft.v1.GraphSnapshot)
                    return object;
                let message = new $root.flowcraft.v1.GraphSnapshot();
                if (object.nodes) {
                    if (!Array.isArray(object.nodes))
                        throw TypeError(".flowcraft.v1.GraphSnapshot.nodes: array expected");
                    message.nodes = [];
                    for (let i = 0; i < object.nodes.length; ++i) {
                        if (typeof object.nodes[i] !== "object")
                            throw TypeError(".flowcraft.v1.GraphSnapshot.nodes: object expected");
                        message.nodes[i] = $root.flowcraft.v1.Node.fromObject(object.nodes[i]);
                    }
                }
                if (object.edges) {
                    if (!Array.isArray(object.edges))
                        throw TypeError(".flowcraft.v1.GraphSnapshot.edges: array expected");
                    message.edges = [];
                    for (let i = 0; i < object.edges.length; ++i) {
                        if (typeof object.edges[i] !== "object")
                            throw TypeError(".flowcraft.v1.GraphSnapshot.edges: object expected");
                        message.edges[i] = $root.flowcraft.v1.Edge.fromObject(object.edges[i]);
                    }
                }
                if (object.version != null)
                    if ($util.Long)
                        (message.version = $util.Long.fromValue(object.version)).unsigned = false;
                    else if (typeof object.version === "string")
                        message.version = parseInt(object.version, 10);
                    else if (typeof object.version === "number")
                        message.version = object.version;
                    else if (typeof object.version === "object")
                        message.version = new $util.LongBits(object.version.low >>> 0, object.version.high >>> 0).toNumber();
                return message;
            };

            /**
             * Creates a plain object from a GraphSnapshot message. Also converts values to other types if specified.
             * @function toObject
             * @memberof flowcraft.v1.GraphSnapshot
             * @static
             * @param {flowcraft.v1.GraphSnapshot} message GraphSnapshot
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            GraphSnapshot.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                let object = {};
                if (options.arrays || options.defaults) {
                    object.nodes = [];
                    object.edges = [];
                }
                if (options.defaults)
                    if ($util.Long) {
                        let long = new $util.Long(0, 0, false);
                        object.version = options.longs === String ? long.toString() : options.longs === Number ? long.toNumber() : long;
                    } else
                        object.version = options.longs === String ? "0" : 0;
                if (message.nodes && message.nodes.length) {
                    object.nodes = [];
                    for (let j = 0; j < message.nodes.length; ++j)
                        object.nodes[j] = $root.flowcraft.v1.Node.toObject(message.nodes[j], options);
                }
                if (message.edges && message.edges.length) {
                    object.edges = [];
                    for (let j = 0; j < message.edges.length; ++j)
                        object.edges[j] = $root.flowcraft.v1.Edge.toObject(message.edges[j], options);
                }
                if (message.version != null && message.hasOwnProperty("version"))
                    if (typeof message.version === "number")
                        object.version = options.longs === String ? String(message.version) : message.version;
                    else
                        object.version = options.longs === String ? $util.Long.prototype.toString.call(message.version) : options.longs === Number ? new $util.LongBits(message.version.low >>> 0, message.version.high >>> 0).toNumber() : message.version;
                return object;
            };

            /**
             * Converts this GraphSnapshot to JSON.
             * @function toJSON
             * @memberof flowcraft.v1.GraphSnapshot
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            GraphSnapshot.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };

            /**
             * Gets the default type url for GraphSnapshot
             * @function getTypeUrl
             * @memberof flowcraft.v1.GraphSnapshot
             * @static
             * @param {string} [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns {string} The default type url
             */
            GraphSnapshot.getTypeUrl = function getTypeUrl(typeUrlPrefix) {
                if (typeUrlPrefix === undefined) {
                    typeUrlPrefix = "type.googleapis.com";
                }
                return typeUrlPrefix + "/flowcraft.v1.GraphSnapshot";
            };

            return GraphSnapshot;
        })();

        v1.GraphMutation = (function() {

            /**
             * Properties of a GraphMutation.
             * @memberof flowcraft.v1
             * @interface IGraphMutation
             * @property {flowcraft.v1.IAddNode|null} [addNode] GraphMutation addNode
             * @property {flowcraft.v1.IUpdateNode|null} [updateNode] GraphMutation updateNode
             * @property {flowcraft.v1.IRemoveNode|null} [removeNode] GraphMutation removeNode
             * @property {flowcraft.v1.IAddEdge|null} [addEdge] GraphMutation addEdge
             * @property {flowcraft.v1.IRemoveEdge|null} [removeEdge] GraphMutation removeEdge
             * @property {flowcraft.v1.IAddSubGraph|null} [addSubgraph] GraphMutation addSubgraph
             * @property {flowcraft.v1.IClearGraph|null} [clearGraph] GraphMutation clearGraph
             */

            /**
             * Constructs a new GraphMutation.
             * @memberof flowcraft.v1
             * @classdesc Represents a GraphMutation.
             * @implements IGraphMutation
             * @constructor
             * @param {flowcraft.v1.IGraphMutation=} [properties] Properties to set
             */
            function GraphMutation(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * GraphMutation addNode.
             * @member {flowcraft.v1.IAddNode|null|undefined} addNode
             * @memberof flowcraft.v1.GraphMutation
             * @instance
             */
            GraphMutation.prototype.addNode = null;

            /**
             * GraphMutation updateNode.
             * @member {flowcraft.v1.IUpdateNode|null|undefined} updateNode
             * @memberof flowcraft.v1.GraphMutation
             * @instance
             */
            GraphMutation.prototype.updateNode = null;

            /**
             * GraphMutation removeNode.
             * @member {flowcraft.v1.IRemoveNode|null|undefined} removeNode
             * @memberof flowcraft.v1.GraphMutation
             * @instance
             */
            GraphMutation.prototype.removeNode = null;

            /**
             * GraphMutation addEdge.
             * @member {flowcraft.v1.IAddEdge|null|undefined} addEdge
             * @memberof flowcraft.v1.GraphMutation
             * @instance
             */
            GraphMutation.prototype.addEdge = null;

            /**
             * GraphMutation removeEdge.
             * @member {flowcraft.v1.IRemoveEdge|null|undefined} removeEdge
             * @memberof flowcraft.v1.GraphMutation
             * @instance
             */
            GraphMutation.prototype.removeEdge = null;

            /**
             * GraphMutation addSubgraph.
             * @member {flowcraft.v1.IAddSubGraph|null|undefined} addSubgraph
             * @memberof flowcraft.v1.GraphMutation
             * @instance
             */
            GraphMutation.prototype.addSubgraph = null;

            /**
             * GraphMutation clearGraph.
             * @member {flowcraft.v1.IClearGraph|null|undefined} clearGraph
             * @memberof flowcraft.v1.GraphMutation
             * @instance
             */
            GraphMutation.prototype.clearGraph = null;

            // OneOf field names bound to virtual getters and setters
            let $oneOfFields;

            /**
             * GraphMutation operation.
             * @member {"addNode"|"updateNode"|"removeNode"|"addEdge"|"removeEdge"|"addSubgraph"|"clearGraph"|undefined} operation
             * @memberof flowcraft.v1.GraphMutation
             * @instance
             */
            Object.defineProperty(GraphMutation.prototype, "operation", {
                get: $util.oneOfGetter($oneOfFields = ["addNode", "updateNode", "removeNode", "addEdge", "removeEdge", "addSubgraph", "clearGraph"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * Creates a new GraphMutation instance using the specified properties.
             * @function create
             * @memberof flowcraft.v1.GraphMutation
             * @static
             * @param {flowcraft.v1.IGraphMutation=} [properties] Properties to set
             * @returns {flowcraft.v1.GraphMutation} GraphMutation instance
             */
            GraphMutation.create = function create(properties) {
                return new GraphMutation(properties);
            };

            /**
             * Encodes the specified GraphMutation message. Does not implicitly {@link flowcraft.v1.GraphMutation.verify|verify} messages.
             * @function encode
             * @memberof flowcraft.v1.GraphMutation
             * @static
             * @param {flowcraft.v1.IGraphMutation} message GraphMutation message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            GraphMutation.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.addNode != null && Object.hasOwnProperty.call(message, "addNode"))
                    $root.flowcraft.v1.AddNode.encode(message.addNode, writer.uint32(/* id 1, wireType 2 =*/10).fork()).ldelim();
                if (message.updateNode != null && Object.hasOwnProperty.call(message, "updateNode"))
                    $root.flowcraft.v1.UpdateNode.encode(message.updateNode, writer.uint32(/* id 2, wireType 2 =*/18).fork()).ldelim();
                if (message.removeNode != null && Object.hasOwnProperty.call(message, "removeNode"))
                    $root.flowcraft.v1.RemoveNode.encode(message.removeNode, writer.uint32(/* id 3, wireType 2 =*/26).fork()).ldelim();
                if (message.addEdge != null && Object.hasOwnProperty.call(message, "addEdge"))
                    $root.flowcraft.v1.AddEdge.encode(message.addEdge, writer.uint32(/* id 4, wireType 2 =*/34).fork()).ldelim();
                if (message.removeEdge != null && Object.hasOwnProperty.call(message, "removeEdge"))
                    $root.flowcraft.v1.RemoveEdge.encode(message.removeEdge, writer.uint32(/* id 5, wireType 2 =*/42).fork()).ldelim();
                if (message.addSubgraph != null && Object.hasOwnProperty.call(message, "addSubgraph"))
                    $root.flowcraft.v1.AddSubGraph.encode(message.addSubgraph, writer.uint32(/* id 6, wireType 2 =*/50).fork()).ldelim();
                if (message.clearGraph != null && Object.hasOwnProperty.call(message, "clearGraph"))
                    $root.flowcraft.v1.ClearGraph.encode(message.clearGraph, writer.uint32(/* id 7, wireType 2 =*/58).fork()).ldelim();
                return writer;
            };

            /**
             * Encodes the specified GraphMutation message, length delimited. Does not implicitly {@link flowcraft.v1.GraphMutation.verify|verify} messages.
             * @function encodeDelimited
             * @memberof flowcraft.v1.GraphMutation
             * @static
             * @param {flowcraft.v1.IGraphMutation} message GraphMutation message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            GraphMutation.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes a GraphMutation message from the specified reader or buffer.
             * @function decode
             * @memberof flowcraft.v1.GraphMutation
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {flowcraft.v1.GraphMutation} GraphMutation
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            GraphMutation.decode = function decode(reader, length, error) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.flowcraft.v1.GraphMutation();
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    if (tag === error)
                        break;
                    switch (tag >>> 3) {
                    case 1: {
                            message.addNode = $root.flowcraft.v1.AddNode.decode(reader, reader.uint32());
                            break;
                        }
                    case 2: {
                            message.updateNode = $root.flowcraft.v1.UpdateNode.decode(reader, reader.uint32());
                            break;
                        }
                    case 3: {
                            message.removeNode = $root.flowcraft.v1.RemoveNode.decode(reader, reader.uint32());
                            break;
                        }
                    case 4: {
                            message.addEdge = $root.flowcraft.v1.AddEdge.decode(reader, reader.uint32());
                            break;
                        }
                    case 5: {
                            message.removeEdge = $root.flowcraft.v1.RemoveEdge.decode(reader, reader.uint32());
                            break;
                        }
                    case 6: {
                            message.addSubgraph = $root.flowcraft.v1.AddSubGraph.decode(reader, reader.uint32());
                            break;
                        }
                    case 7: {
                            message.clearGraph = $root.flowcraft.v1.ClearGraph.decode(reader, reader.uint32());
                            break;
                        }
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };

            /**
             * Decodes a GraphMutation message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof flowcraft.v1.GraphMutation
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {flowcraft.v1.GraphMutation} GraphMutation
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            GraphMutation.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies a GraphMutation message.
             * @function verify
             * @memberof flowcraft.v1.GraphMutation
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            GraphMutation.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                let properties = {};
                if (message.addNode != null && message.hasOwnProperty("addNode")) {
                    properties.operation = 1;
                    {
                        let error = $root.flowcraft.v1.AddNode.verify(message.addNode);
                        if (error)
                            return "addNode." + error;
                    }
                }
                if (message.updateNode != null && message.hasOwnProperty("updateNode")) {
                    if (properties.operation === 1)
                        return "operation: multiple values";
                    properties.operation = 1;
                    {
                        let error = $root.flowcraft.v1.UpdateNode.verify(message.updateNode);
                        if (error)
                            return "updateNode." + error;
                    }
                }
                if (message.removeNode != null && message.hasOwnProperty("removeNode")) {
                    if (properties.operation === 1)
                        return "operation: multiple values";
                    properties.operation = 1;
                    {
                        let error = $root.flowcraft.v1.RemoveNode.verify(message.removeNode);
                        if (error)
                            return "removeNode." + error;
                    }
                }
                if (message.addEdge != null && message.hasOwnProperty("addEdge")) {
                    if (properties.operation === 1)
                        return "operation: multiple values";
                    properties.operation = 1;
                    {
                        let error = $root.flowcraft.v1.AddEdge.verify(message.addEdge);
                        if (error)
                            return "addEdge." + error;
                    }
                }
                if (message.removeEdge != null && message.hasOwnProperty("removeEdge")) {
                    if (properties.operation === 1)
                        return "operation: multiple values";
                    properties.operation = 1;
                    {
                        let error = $root.flowcraft.v1.RemoveEdge.verify(message.removeEdge);
                        if (error)
                            return "removeEdge." + error;
                    }
                }
                if (message.addSubgraph != null && message.hasOwnProperty("addSubgraph")) {
                    if (properties.operation === 1)
                        return "operation: multiple values";
                    properties.operation = 1;
                    {
                        let error = $root.flowcraft.v1.AddSubGraph.verify(message.addSubgraph);
                        if (error)
                            return "addSubgraph." + error;
                    }
                }
                if (message.clearGraph != null && message.hasOwnProperty("clearGraph")) {
                    if (properties.operation === 1)
                        return "operation: multiple values";
                    properties.operation = 1;
                    {
                        let error = $root.flowcraft.v1.ClearGraph.verify(message.clearGraph);
                        if (error)
                            return "clearGraph." + error;
                    }
                }
                return null;
            };

            /**
             * Creates a GraphMutation message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof flowcraft.v1.GraphMutation
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {flowcraft.v1.GraphMutation} GraphMutation
             */
            GraphMutation.fromObject = function fromObject(object) {
                if (object instanceof $root.flowcraft.v1.GraphMutation)
                    return object;
                let message = new $root.flowcraft.v1.GraphMutation();
                if (object.addNode != null) {
                    if (typeof object.addNode !== "object")
                        throw TypeError(".flowcraft.v1.GraphMutation.addNode: object expected");
                    message.addNode = $root.flowcraft.v1.AddNode.fromObject(object.addNode);
                }
                if (object.updateNode != null) {
                    if (typeof object.updateNode !== "object")
                        throw TypeError(".flowcraft.v1.GraphMutation.updateNode: object expected");
                    message.updateNode = $root.flowcraft.v1.UpdateNode.fromObject(object.updateNode);
                }
                if (object.removeNode != null) {
                    if (typeof object.removeNode !== "object")
                        throw TypeError(".flowcraft.v1.GraphMutation.removeNode: object expected");
                    message.removeNode = $root.flowcraft.v1.RemoveNode.fromObject(object.removeNode);
                }
                if (object.addEdge != null) {
                    if (typeof object.addEdge !== "object")
                        throw TypeError(".flowcraft.v1.GraphMutation.addEdge: object expected");
                    message.addEdge = $root.flowcraft.v1.AddEdge.fromObject(object.addEdge);
                }
                if (object.removeEdge != null) {
                    if (typeof object.removeEdge !== "object")
                        throw TypeError(".flowcraft.v1.GraphMutation.removeEdge: object expected");
                    message.removeEdge = $root.flowcraft.v1.RemoveEdge.fromObject(object.removeEdge);
                }
                if (object.addSubgraph != null) {
                    if (typeof object.addSubgraph !== "object")
                        throw TypeError(".flowcraft.v1.GraphMutation.addSubgraph: object expected");
                    message.addSubgraph = $root.flowcraft.v1.AddSubGraph.fromObject(object.addSubgraph);
                }
                if (object.clearGraph != null) {
                    if (typeof object.clearGraph !== "object")
                        throw TypeError(".flowcraft.v1.GraphMutation.clearGraph: object expected");
                    message.clearGraph = $root.flowcraft.v1.ClearGraph.fromObject(object.clearGraph);
                }
                return message;
            };

            /**
             * Creates a plain object from a GraphMutation message. Also converts values to other types if specified.
             * @function toObject
             * @memberof flowcraft.v1.GraphMutation
             * @static
             * @param {flowcraft.v1.GraphMutation} message GraphMutation
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            GraphMutation.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                let object = {};
                if (message.addNode != null && message.hasOwnProperty("addNode")) {
                    object.addNode = $root.flowcraft.v1.AddNode.toObject(message.addNode, options);
                    if (options.oneofs)
                        object.operation = "addNode";
                }
                if (message.updateNode != null && message.hasOwnProperty("updateNode")) {
                    object.updateNode = $root.flowcraft.v1.UpdateNode.toObject(message.updateNode, options);
                    if (options.oneofs)
                        object.operation = "updateNode";
                }
                if (message.removeNode != null && message.hasOwnProperty("removeNode")) {
                    object.removeNode = $root.flowcraft.v1.RemoveNode.toObject(message.removeNode, options);
                    if (options.oneofs)
                        object.operation = "removeNode";
                }
                if (message.addEdge != null && message.hasOwnProperty("addEdge")) {
                    object.addEdge = $root.flowcraft.v1.AddEdge.toObject(message.addEdge, options);
                    if (options.oneofs)
                        object.operation = "addEdge";
                }
                if (message.removeEdge != null && message.hasOwnProperty("removeEdge")) {
                    object.removeEdge = $root.flowcraft.v1.RemoveEdge.toObject(message.removeEdge, options);
                    if (options.oneofs)
                        object.operation = "removeEdge";
                }
                if (message.addSubgraph != null && message.hasOwnProperty("addSubgraph")) {
                    object.addSubgraph = $root.flowcraft.v1.AddSubGraph.toObject(message.addSubgraph, options);
                    if (options.oneofs)
                        object.operation = "addSubgraph";
                }
                if (message.clearGraph != null && message.hasOwnProperty("clearGraph")) {
                    object.clearGraph = $root.flowcraft.v1.ClearGraph.toObject(message.clearGraph, options);
                    if (options.oneofs)
                        object.operation = "clearGraph";
                }
                return object;
            };

            /**
             * Converts this GraphMutation to JSON.
             * @function toJSON
             * @memberof flowcraft.v1.GraphMutation
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            GraphMutation.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };

            /**
             * Gets the default type url for GraphMutation
             * @function getTypeUrl
             * @memberof flowcraft.v1.GraphMutation
             * @static
             * @param {string} [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns {string} The default type url
             */
            GraphMutation.getTypeUrl = function getTypeUrl(typeUrlPrefix) {
                if (typeUrlPrefix === undefined) {
                    typeUrlPrefix = "type.googleapis.com";
                }
                return typeUrlPrefix + "/flowcraft.v1.GraphMutation";
            };

            return GraphMutation;
        })();

        v1.AddNode = (function() {

            /**
             * Properties of an AddNode.
             * @memberof flowcraft.v1
             * @interface IAddNode
             * @property {flowcraft.v1.INode|null} [node] AddNode node
             */

            /**
             * Constructs a new AddNode.
             * @memberof flowcraft.v1
             * @classdesc Represents an AddNode.
             * @implements IAddNode
             * @constructor
             * @param {flowcraft.v1.IAddNode=} [properties] Properties to set
             */
            function AddNode(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * AddNode node.
             * @member {flowcraft.v1.INode|null|undefined} node
             * @memberof flowcraft.v1.AddNode
             * @instance
             */
            AddNode.prototype.node = null;

            /**
             * Creates a new AddNode instance using the specified properties.
             * @function create
             * @memberof flowcraft.v1.AddNode
             * @static
             * @param {flowcraft.v1.IAddNode=} [properties] Properties to set
             * @returns {flowcraft.v1.AddNode} AddNode instance
             */
            AddNode.create = function create(properties) {
                return new AddNode(properties);
            };

            /**
             * Encodes the specified AddNode message. Does not implicitly {@link flowcraft.v1.AddNode.verify|verify} messages.
             * @function encode
             * @memberof flowcraft.v1.AddNode
             * @static
             * @param {flowcraft.v1.IAddNode} message AddNode message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            AddNode.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.node != null && Object.hasOwnProperty.call(message, "node"))
                    $root.flowcraft.v1.Node.encode(message.node, writer.uint32(/* id 1, wireType 2 =*/10).fork()).ldelim();
                return writer;
            };

            /**
             * Encodes the specified AddNode message, length delimited. Does not implicitly {@link flowcraft.v1.AddNode.verify|verify} messages.
             * @function encodeDelimited
             * @memberof flowcraft.v1.AddNode
             * @static
             * @param {flowcraft.v1.IAddNode} message AddNode message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            AddNode.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes an AddNode message from the specified reader or buffer.
             * @function decode
             * @memberof flowcraft.v1.AddNode
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {flowcraft.v1.AddNode} AddNode
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            AddNode.decode = function decode(reader, length, error) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.flowcraft.v1.AddNode();
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    if (tag === error)
                        break;
                    switch (tag >>> 3) {
                    case 1: {
                            message.node = $root.flowcraft.v1.Node.decode(reader, reader.uint32());
                            break;
                        }
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };

            /**
             * Decodes an AddNode message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof flowcraft.v1.AddNode
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {flowcraft.v1.AddNode} AddNode
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            AddNode.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies an AddNode message.
             * @function verify
             * @memberof flowcraft.v1.AddNode
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            AddNode.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.node != null && message.hasOwnProperty("node")) {
                    let error = $root.flowcraft.v1.Node.verify(message.node);
                    if (error)
                        return "node." + error;
                }
                return null;
            };

            /**
             * Creates an AddNode message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof flowcraft.v1.AddNode
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {flowcraft.v1.AddNode} AddNode
             */
            AddNode.fromObject = function fromObject(object) {
                if (object instanceof $root.flowcraft.v1.AddNode)
                    return object;
                let message = new $root.flowcraft.v1.AddNode();
                if (object.node != null) {
                    if (typeof object.node !== "object")
                        throw TypeError(".flowcraft.v1.AddNode.node: object expected");
                    message.node = $root.flowcraft.v1.Node.fromObject(object.node);
                }
                return message;
            };

            /**
             * Creates a plain object from an AddNode message. Also converts values to other types if specified.
             * @function toObject
             * @memberof flowcraft.v1.AddNode
             * @static
             * @param {flowcraft.v1.AddNode} message AddNode
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            AddNode.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                let object = {};
                if (options.defaults)
                    object.node = null;
                if (message.node != null && message.hasOwnProperty("node"))
                    object.node = $root.flowcraft.v1.Node.toObject(message.node, options);
                return object;
            };

            /**
             * Converts this AddNode to JSON.
             * @function toJSON
             * @memberof flowcraft.v1.AddNode
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            AddNode.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };

            /**
             * Gets the default type url for AddNode
             * @function getTypeUrl
             * @memberof flowcraft.v1.AddNode
             * @static
             * @param {string} [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns {string} The default type url
             */
            AddNode.getTypeUrl = function getTypeUrl(typeUrlPrefix) {
                if (typeUrlPrefix === undefined) {
                    typeUrlPrefix = "type.googleapis.com";
                }
                return typeUrlPrefix + "/flowcraft.v1.AddNode";
            };

            return AddNode;
        })();

        v1.UpdateNode = (function() {

            /**
             * Properties of an UpdateNode.
             * @memberof flowcraft.v1
             * @interface IUpdateNode
             * @property {string|null} [id] UpdateNode id
             * @property {flowcraft.v1.INodeData|null} [data] UpdateNode data
             * @property {flowcraft.v1.IPosition|null} [position] UpdateNode position
             */

            /**
             * Constructs a new UpdateNode.
             * @memberof flowcraft.v1
             * @classdesc Represents an UpdateNode.
             * @implements IUpdateNode
             * @constructor
             * @param {flowcraft.v1.IUpdateNode=} [properties] Properties to set
             */
            function UpdateNode(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * UpdateNode id.
             * @member {string} id
             * @memberof flowcraft.v1.UpdateNode
             * @instance
             */
            UpdateNode.prototype.id = "";

            /**
             * UpdateNode data.
             * @member {flowcraft.v1.INodeData|null|undefined} data
             * @memberof flowcraft.v1.UpdateNode
             * @instance
             */
            UpdateNode.prototype.data = null;

            /**
             * UpdateNode position.
             * @member {flowcraft.v1.IPosition|null|undefined} position
             * @memberof flowcraft.v1.UpdateNode
             * @instance
             */
            UpdateNode.prototype.position = null;

            /**
             * Creates a new UpdateNode instance using the specified properties.
             * @function create
             * @memberof flowcraft.v1.UpdateNode
             * @static
             * @param {flowcraft.v1.IUpdateNode=} [properties] Properties to set
             * @returns {flowcraft.v1.UpdateNode} UpdateNode instance
             */
            UpdateNode.create = function create(properties) {
                return new UpdateNode(properties);
            };

            /**
             * Encodes the specified UpdateNode message. Does not implicitly {@link flowcraft.v1.UpdateNode.verify|verify} messages.
             * @function encode
             * @memberof flowcraft.v1.UpdateNode
             * @static
             * @param {flowcraft.v1.IUpdateNode} message UpdateNode message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            UpdateNode.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.id != null && Object.hasOwnProperty.call(message, "id"))
                    writer.uint32(/* id 1, wireType 2 =*/10).string(message.id);
                if (message.data != null && Object.hasOwnProperty.call(message, "data"))
                    $root.flowcraft.v1.NodeData.encode(message.data, writer.uint32(/* id 2, wireType 2 =*/18).fork()).ldelim();
                if (message.position != null && Object.hasOwnProperty.call(message, "position"))
                    $root.flowcraft.v1.Position.encode(message.position, writer.uint32(/* id 3, wireType 2 =*/26).fork()).ldelim();
                return writer;
            };

            /**
             * Encodes the specified UpdateNode message, length delimited. Does not implicitly {@link flowcraft.v1.UpdateNode.verify|verify} messages.
             * @function encodeDelimited
             * @memberof flowcraft.v1.UpdateNode
             * @static
             * @param {flowcraft.v1.IUpdateNode} message UpdateNode message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            UpdateNode.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes an UpdateNode message from the specified reader or buffer.
             * @function decode
             * @memberof flowcraft.v1.UpdateNode
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {flowcraft.v1.UpdateNode} UpdateNode
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            UpdateNode.decode = function decode(reader, length, error) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.flowcraft.v1.UpdateNode();
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    if (tag === error)
                        break;
                    switch (tag >>> 3) {
                    case 1: {
                            message.id = reader.string();
                            break;
                        }
                    case 2: {
                            message.data = $root.flowcraft.v1.NodeData.decode(reader, reader.uint32());
                            break;
                        }
                    case 3: {
                            message.position = $root.flowcraft.v1.Position.decode(reader, reader.uint32());
                            break;
                        }
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };

            /**
             * Decodes an UpdateNode message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof flowcraft.v1.UpdateNode
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {flowcraft.v1.UpdateNode} UpdateNode
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            UpdateNode.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies an UpdateNode message.
             * @function verify
             * @memberof flowcraft.v1.UpdateNode
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            UpdateNode.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.id != null && message.hasOwnProperty("id"))
                    if (!$util.isString(message.id))
                        return "id: string expected";
                if (message.data != null && message.hasOwnProperty("data")) {
                    let error = $root.flowcraft.v1.NodeData.verify(message.data);
                    if (error)
                        return "data." + error;
                }
                if (message.position != null && message.hasOwnProperty("position")) {
                    let error = $root.flowcraft.v1.Position.verify(message.position);
                    if (error)
                        return "position." + error;
                }
                return null;
            };

            /**
             * Creates an UpdateNode message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof flowcraft.v1.UpdateNode
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {flowcraft.v1.UpdateNode} UpdateNode
             */
            UpdateNode.fromObject = function fromObject(object) {
                if (object instanceof $root.flowcraft.v1.UpdateNode)
                    return object;
                let message = new $root.flowcraft.v1.UpdateNode();
                if (object.id != null)
                    message.id = String(object.id);
                if (object.data != null) {
                    if (typeof object.data !== "object")
                        throw TypeError(".flowcraft.v1.UpdateNode.data: object expected");
                    message.data = $root.flowcraft.v1.NodeData.fromObject(object.data);
                }
                if (object.position != null) {
                    if (typeof object.position !== "object")
                        throw TypeError(".flowcraft.v1.UpdateNode.position: object expected");
                    message.position = $root.flowcraft.v1.Position.fromObject(object.position);
                }
                return message;
            };

            /**
             * Creates a plain object from an UpdateNode message. Also converts values to other types if specified.
             * @function toObject
             * @memberof flowcraft.v1.UpdateNode
             * @static
             * @param {flowcraft.v1.UpdateNode} message UpdateNode
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            UpdateNode.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                let object = {};
                if (options.defaults) {
                    object.id = "";
                    object.data = null;
                    object.position = null;
                }
                if (message.id != null && message.hasOwnProperty("id"))
                    object.id = message.id;
                if (message.data != null && message.hasOwnProperty("data"))
                    object.data = $root.flowcraft.v1.NodeData.toObject(message.data, options);
                if (message.position != null && message.hasOwnProperty("position"))
                    object.position = $root.flowcraft.v1.Position.toObject(message.position, options);
                return object;
            };

            /**
             * Converts this UpdateNode to JSON.
             * @function toJSON
             * @memberof flowcraft.v1.UpdateNode
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            UpdateNode.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };

            /**
             * Gets the default type url for UpdateNode
             * @function getTypeUrl
             * @memberof flowcraft.v1.UpdateNode
             * @static
             * @param {string} [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns {string} The default type url
             */
            UpdateNode.getTypeUrl = function getTypeUrl(typeUrlPrefix) {
                if (typeUrlPrefix === undefined) {
                    typeUrlPrefix = "type.googleapis.com";
                }
                return typeUrlPrefix + "/flowcraft.v1.UpdateNode";
            };

            return UpdateNode;
        })();

        v1.RemoveNode = (function() {

            /**
             * Properties of a RemoveNode.
             * @memberof flowcraft.v1
             * @interface IRemoveNode
             * @property {string|null} [id] RemoveNode id
             */

            /**
             * Constructs a new RemoveNode.
             * @memberof flowcraft.v1
             * @classdesc Represents a RemoveNode.
             * @implements IRemoveNode
             * @constructor
             * @param {flowcraft.v1.IRemoveNode=} [properties] Properties to set
             */
            function RemoveNode(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * RemoveNode id.
             * @member {string} id
             * @memberof flowcraft.v1.RemoveNode
             * @instance
             */
            RemoveNode.prototype.id = "";

            /**
             * Creates a new RemoveNode instance using the specified properties.
             * @function create
             * @memberof flowcraft.v1.RemoveNode
             * @static
             * @param {flowcraft.v1.IRemoveNode=} [properties] Properties to set
             * @returns {flowcraft.v1.RemoveNode} RemoveNode instance
             */
            RemoveNode.create = function create(properties) {
                return new RemoveNode(properties);
            };

            /**
             * Encodes the specified RemoveNode message. Does not implicitly {@link flowcraft.v1.RemoveNode.verify|verify} messages.
             * @function encode
             * @memberof flowcraft.v1.RemoveNode
             * @static
             * @param {flowcraft.v1.IRemoveNode} message RemoveNode message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            RemoveNode.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.id != null && Object.hasOwnProperty.call(message, "id"))
                    writer.uint32(/* id 1, wireType 2 =*/10).string(message.id);
                return writer;
            };

            /**
             * Encodes the specified RemoveNode message, length delimited. Does not implicitly {@link flowcraft.v1.RemoveNode.verify|verify} messages.
             * @function encodeDelimited
             * @memberof flowcraft.v1.RemoveNode
             * @static
             * @param {flowcraft.v1.IRemoveNode} message RemoveNode message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            RemoveNode.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes a RemoveNode message from the specified reader or buffer.
             * @function decode
             * @memberof flowcraft.v1.RemoveNode
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {flowcraft.v1.RemoveNode} RemoveNode
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            RemoveNode.decode = function decode(reader, length, error) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.flowcraft.v1.RemoveNode();
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    if (tag === error)
                        break;
                    switch (tag >>> 3) {
                    case 1: {
                            message.id = reader.string();
                            break;
                        }
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };

            /**
             * Decodes a RemoveNode message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof flowcraft.v1.RemoveNode
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {flowcraft.v1.RemoveNode} RemoveNode
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            RemoveNode.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies a RemoveNode message.
             * @function verify
             * @memberof flowcraft.v1.RemoveNode
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            RemoveNode.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.id != null && message.hasOwnProperty("id"))
                    if (!$util.isString(message.id))
                        return "id: string expected";
                return null;
            };

            /**
             * Creates a RemoveNode message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof flowcraft.v1.RemoveNode
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {flowcraft.v1.RemoveNode} RemoveNode
             */
            RemoveNode.fromObject = function fromObject(object) {
                if (object instanceof $root.flowcraft.v1.RemoveNode)
                    return object;
                let message = new $root.flowcraft.v1.RemoveNode();
                if (object.id != null)
                    message.id = String(object.id);
                return message;
            };

            /**
             * Creates a plain object from a RemoveNode message. Also converts values to other types if specified.
             * @function toObject
             * @memberof flowcraft.v1.RemoveNode
             * @static
             * @param {flowcraft.v1.RemoveNode} message RemoveNode
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            RemoveNode.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                let object = {};
                if (options.defaults)
                    object.id = "";
                if (message.id != null && message.hasOwnProperty("id"))
                    object.id = message.id;
                return object;
            };

            /**
             * Converts this RemoveNode to JSON.
             * @function toJSON
             * @memberof flowcraft.v1.RemoveNode
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            RemoveNode.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };

            /**
             * Gets the default type url for RemoveNode
             * @function getTypeUrl
             * @memberof flowcraft.v1.RemoveNode
             * @static
             * @param {string} [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns {string} The default type url
             */
            RemoveNode.getTypeUrl = function getTypeUrl(typeUrlPrefix) {
                if (typeUrlPrefix === undefined) {
                    typeUrlPrefix = "type.googleapis.com";
                }
                return typeUrlPrefix + "/flowcraft.v1.RemoveNode";
            };

            return RemoveNode;
        })();

        v1.AddEdge = (function() {

            /**
             * Properties of an AddEdge.
             * @memberof flowcraft.v1
             * @interface IAddEdge
             * @property {flowcraft.v1.IEdge|null} [edge] AddEdge edge
             */

            /**
             * Constructs a new AddEdge.
             * @memberof flowcraft.v1
             * @classdesc Represents an AddEdge.
             * @implements IAddEdge
             * @constructor
             * @param {flowcraft.v1.IAddEdge=} [properties] Properties to set
             */
            function AddEdge(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * AddEdge edge.
             * @member {flowcraft.v1.IEdge|null|undefined} edge
             * @memberof flowcraft.v1.AddEdge
             * @instance
             */
            AddEdge.prototype.edge = null;

            /**
             * Creates a new AddEdge instance using the specified properties.
             * @function create
             * @memberof flowcraft.v1.AddEdge
             * @static
             * @param {flowcraft.v1.IAddEdge=} [properties] Properties to set
             * @returns {flowcraft.v1.AddEdge} AddEdge instance
             */
            AddEdge.create = function create(properties) {
                return new AddEdge(properties);
            };

            /**
             * Encodes the specified AddEdge message. Does not implicitly {@link flowcraft.v1.AddEdge.verify|verify} messages.
             * @function encode
             * @memberof flowcraft.v1.AddEdge
             * @static
             * @param {flowcraft.v1.IAddEdge} message AddEdge message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            AddEdge.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.edge != null && Object.hasOwnProperty.call(message, "edge"))
                    $root.flowcraft.v1.Edge.encode(message.edge, writer.uint32(/* id 1, wireType 2 =*/10).fork()).ldelim();
                return writer;
            };

            /**
             * Encodes the specified AddEdge message, length delimited. Does not implicitly {@link flowcraft.v1.AddEdge.verify|verify} messages.
             * @function encodeDelimited
             * @memberof flowcraft.v1.AddEdge
             * @static
             * @param {flowcraft.v1.IAddEdge} message AddEdge message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            AddEdge.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes an AddEdge message from the specified reader or buffer.
             * @function decode
             * @memberof flowcraft.v1.AddEdge
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {flowcraft.v1.AddEdge} AddEdge
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            AddEdge.decode = function decode(reader, length, error) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.flowcraft.v1.AddEdge();
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    if (tag === error)
                        break;
                    switch (tag >>> 3) {
                    case 1: {
                            message.edge = $root.flowcraft.v1.Edge.decode(reader, reader.uint32());
                            break;
                        }
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };

            /**
             * Decodes an AddEdge message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof flowcraft.v1.AddEdge
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {flowcraft.v1.AddEdge} AddEdge
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            AddEdge.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies an AddEdge message.
             * @function verify
             * @memberof flowcraft.v1.AddEdge
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            AddEdge.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.edge != null && message.hasOwnProperty("edge")) {
                    let error = $root.flowcraft.v1.Edge.verify(message.edge);
                    if (error)
                        return "edge." + error;
                }
                return null;
            };

            /**
             * Creates an AddEdge message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof flowcraft.v1.AddEdge
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {flowcraft.v1.AddEdge} AddEdge
             */
            AddEdge.fromObject = function fromObject(object) {
                if (object instanceof $root.flowcraft.v1.AddEdge)
                    return object;
                let message = new $root.flowcraft.v1.AddEdge();
                if (object.edge != null) {
                    if (typeof object.edge !== "object")
                        throw TypeError(".flowcraft.v1.AddEdge.edge: object expected");
                    message.edge = $root.flowcraft.v1.Edge.fromObject(object.edge);
                }
                return message;
            };

            /**
             * Creates a plain object from an AddEdge message. Also converts values to other types if specified.
             * @function toObject
             * @memberof flowcraft.v1.AddEdge
             * @static
             * @param {flowcraft.v1.AddEdge} message AddEdge
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            AddEdge.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                let object = {};
                if (options.defaults)
                    object.edge = null;
                if (message.edge != null && message.hasOwnProperty("edge"))
                    object.edge = $root.flowcraft.v1.Edge.toObject(message.edge, options);
                return object;
            };

            /**
             * Converts this AddEdge to JSON.
             * @function toJSON
             * @memberof flowcraft.v1.AddEdge
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            AddEdge.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };

            /**
             * Gets the default type url for AddEdge
             * @function getTypeUrl
             * @memberof flowcraft.v1.AddEdge
             * @static
             * @param {string} [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns {string} The default type url
             */
            AddEdge.getTypeUrl = function getTypeUrl(typeUrlPrefix) {
                if (typeUrlPrefix === undefined) {
                    typeUrlPrefix = "type.googleapis.com";
                }
                return typeUrlPrefix + "/flowcraft.v1.AddEdge";
            };

            return AddEdge;
        })();

        v1.RemoveEdge = (function() {

            /**
             * Properties of a RemoveEdge.
             * @memberof flowcraft.v1
             * @interface IRemoveEdge
             * @property {string|null} [id] RemoveEdge id
             */

            /**
             * Constructs a new RemoveEdge.
             * @memberof flowcraft.v1
             * @classdesc Represents a RemoveEdge.
             * @implements IRemoveEdge
             * @constructor
             * @param {flowcraft.v1.IRemoveEdge=} [properties] Properties to set
             */
            function RemoveEdge(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * RemoveEdge id.
             * @member {string} id
             * @memberof flowcraft.v1.RemoveEdge
             * @instance
             */
            RemoveEdge.prototype.id = "";

            /**
             * Creates a new RemoveEdge instance using the specified properties.
             * @function create
             * @memberof flowcraft.v1.RemoveEdge
             * @static
             * @param {flowcraft.v1.IRemoveEdge=} [properties] Properties to set
             * @returns {flowcraft.v1.RemoveEdge} RemoveEdge instance
             */
            RemoveEdge.create = function create(properties) {
                return new RemoveEdge(properties);
            };

            /**
             * Encodes the specified RemoveEdge message. Does not implicitly {@link flowcraft.v1.RemoveEdge.verify|verify} messages.
             * @function encode
             * @memberof flowcraft.v1.RemoveEdge
             * @static
             * @param {flowcraft.v1.IRemoveEdge} message RemoveEdge message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            RemoveEdge.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.id != null && Object.hasOwnProperty.call(message, "id"))
                    writer.uint32(/* id 1, wireType 2 =*/10).string(message.id);
                return writer;
            };

            /**
             * Encodes the specified RemoveEdge message, length delimited. Does not implicitly {@link flowcraft.v1.RemoveEdge.verify|verify} messages.
             * @function encodeDelimited
             * @memberof flowcraft.v1.RemoveEdge
             * @static
             * @param {flowcraft.v1.IRemoveEdge} message RemoveEdge message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            RemoveEdge.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes a RemoveEdge message from the specified reader or buffer.
             * @function decode
             * @memberof flowcraft.v1.RemoveEdge
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {flowcraft.v1.RemoveEdge} RemoveEdge
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            RemoveEdge.decode = function decode(reader, length, error) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.flowcraft.v1.RemoveEdge();
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    if (tag === error)
                        break;
                    switch (tag >>> 3) {
                    case 1: {
                            message.id = reader.string();
                            break;
                        }
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };

            /**
             * Decodes a RemoveEdge message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof flowcraft.v1.RemoveEdge
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {flowcraft.v1.RemoveEdge} RemoveEdge
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            RemoveEdge.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies a RemoveEdge message.
             * @function verify
             * @memberof flowcraft.v1.RemoveEdge
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            RemoveEdge.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.id != null && message.hasOwnProperty("id"))
                    if (!$util.isString(message.id))
                        return "id: string expected";
                return null;
            };

            /**
             * Creates a RemoveEdge message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof flowcraft.v1.RemoveEdge
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {flowcraft.v1.RemoveEdge} RemoveEdge
             */
            RemoveEdge.fromObject = function fromObject(object) {
                if (object instanceof $root.flowcraft.v1.RemoveEdge)
                    return object;
                let message = new $root.flowcraft.v1.RemoveEdge();
                if (object.id != null)
                    message.id = String(object.id);
                return message;
            };

            /**
             * Creates a plain object from a RemoveEdge message. Also converts values to other types if specified.
             * @function toObject
             * @memberof flowcraft.v1.RemoveEdge
             * @static
             * @param {flowcraft.v1.RemoveEdge} message RemoveEdge
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            RemoveEdge.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                let object = {};
                if (options.defaults)
                    object.id = "";
                if (message.id != null && message.hasOwnProperty("id"))
                    object.id = message.id;
                return object;
            };

            /**
             * Converts this RemoveEdge to JSON.
             * @function toJSON
             * @memberof flowcraft.v1.RemoveEdge
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            RemoveEdge.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };

            /**
             * Gets the default type url for RemoveEdge
             * @function getTypeUrl
             * @memberof flowcraft.v1.RemoveEdge
             * @static
             * @param {string} [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns {string} The default type url
             */
            RemoveEdge.getTypeUrl = function getTypeUrl(typeUrlPrefix) {
                if (typeUrlPrefix === undefined) {
                    typeUrlPrefix = "type.googleapis.com";
                }
                return typeUrlPrefix + "/flowcraft.v1.RemoveEdge";
            };

            return RemoveEdge;
        })();

        v1.AddSubGraph = (function() {

            /**
             * Properties of an AddSubGraph.
             * @memberof flowcraft.v1
             * @interface IAddSubGraph
             * @property {Array.<flowcraft.v1.INode>|null} [nodes] AddSubGraph nodes
             * @property {Array.<flowcraft.v1.IEdge>|null} [edges] AddSubGraph edges
             */

            /**
             * Constructs a new AddSubGraph.
             * @memberof flowcraft.v1
             * @classdesc Represents an AddSubGraph.
             * @implements IAddSubGraph
             * @constructor
             * @param {flowcraft.v1.IAddSubGraph=} [properties] Properties to set
             */
            function AddSubGraph(properties) {
                this.nodes = [];
                this.edges = [];
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * AddSubGraph nodes.
             * @member {Array.<flowcraft.v1.INode>} nodes
             * @memberof flowcraft.v1.AddSubGraph
             * @instance
             */
            AddSubGraph.prototype.nodes = $util.emptyArray;

            /**
             * AddSubGraph edges.
             * @member {Array.<flowcraft.v1.IEdge>} edges
             * @memberof flowcraft.v1.AddSubGraph
             * @instance
             */
            AddSubGraph.prototype.edges = $util.emptyArray;

            /**
             * Creates a new AddSubGraph instance using the specified properties.
             * @function create
             * @memberof flowcraft.v1.AddSubGraph
             * @static
             * @param {flowcraft.v1.IAddSubGraph=} [properties] Properties to set
             * @returns {flowcraft.v1.AddSubGraph} AddSubGraph instance
             */
            AddSubGraph.create = function create(properties) {
                return new AddSubGraph(properties);
            };

            /**
             * Encodes the specified AddSubGraph message. Does not implicitly {@link flowcraft.v1.AddSubGraph.verify|verify} messages.
             * @function encode
             * @memberof flowcraft.v1.AddSubGraph
             * @static
             * @param {flowcraft.v1.IAddSubGraph} message AddSubGraph message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            AddSubGraph.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.nodes != null && message.nodes.length)
                    for (let i = 0; i < message.nodes.length; ++i)
                        $root.flowcraft.v1.Node.encode(message.nodes[i], writer.uint32(/* id 1, wireType 2 =*/10).fork()).ldelim();
                if (message.edges != null && message.edges.length)
                    for (let i = 0; i < message.edges.length; ++i)
                        $root.flowcraft.v1.Edge.encode(message.edges[i], writer.uint32(/* id 2, wireType 2 =*/18).fork()).ldelim();
                return writer;
            };

            /**
             * Encodes the specified AddSubGraph message, length delimited. Does not implicitly {@link flowcraft.v1.AddSubGraph.verify|verify} messages.
             * @function encodeDelimited
             * @memberof flowcraft.v1.AddSubGraph
             * @static
             * @param {flowcraft.v1.IAddSubGraph} message AddSubGraph message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            AddSubGraph.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes an AddSubGraph message from the specified reader or buffer.
             * @function decode
             * @memberof flowcraft.v1.AddSubGraph
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {flowcraft.v1.AddSubGraph} AddSubGraph
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            AddSubGraph.decode = function decode(reader, length, error) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.flowcraft.v1.AddSubGraph();
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    if (tag === error)
                        break;
                    switch (tag >>> 3) {
                    case 1: {
                            if (!(message.nodes && message.nodes.length))
                                message.nodes = [];
                            message.nodes.push($root.flowcraft.v1.Node.decode(reader, reader.uint32()));
                            break;
                        }
                    case 2: {
                            if (!(message.edges && message.edges.length))
                                message.edges = [];
                            message.edges.push($root.flowcraft.v1.Edge.decode(reader, reader.uint32()));
                            break;
                        }
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };

            /**
             * Decodes an AddSubGraph message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof flowcraft.v1.AddSubGraph
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {flowcraft.v1.AddSubGraph} AddSubGraph
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            AddSubGraph.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies an AddSubGraph message.
             * @function verify
             * @memberof flowcraft.v1.AddSubGraph
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            AddSubGraph.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.nodes != null && message.hasOwnProperty("nodes")) {
                    if (!Array.isArray(message.nodes))
                        return "nodes: array expected";
                    for (let i = 0; i < message.nodes.length; ++i) {
                        let error = $root.flowcraft.v1.Node.verify(message.nodes[i]);
                        if (error)
                            return "nodes." + error;
                    }
                }
                if (message.edges != null && message.hasOwnProperty("edges")) {
                    if (!Array.isArray(message.edges))
                        return "edges: array expected";
                    for (let i = 0; i < message.edges.length; ++i) {
                        let error = $root.flowcraft.v1.Edge.verify(message.edges[i]);
                        if (error)
                            return "edges." + error;
                    }
                }
                return null;
            };

            /**
             * Creates an AddSubGraph message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof flowcraft.v1.AddSubGraph
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {flowcraft.v1.AddSubGraph} AddSubGraph
             */
            AddSubGraph.fromObject = function fromObject(object) {
                if (object instanceof $root.flowcraft.v1.AddSubGraph)
                    return object;
                let message = new $root.flowcraft.v1.AddSubGraph();
                if (object.nodes) {
                    if (!Array.isArray(object.nodes))
                        throw TypeError(".flowcraft.v1.AddSubGraph.nodes: array expected");
                    message.nodes = [];
                    for (let i = 0; i < object.nodes.length; ++i) {
                        if (typeof object.nodes[i] !== "object")
                            throw TypeError(".flowcraft.v1.AddSubGraph.nodes: object expected");
                        message.nodes[i] = $root.flowcraft.v1.Node.fromObject(object.nodes[i]);
                    }
                }
                if (object.edges) {
                    if (!Array.isArray(object.edges))
                        throw TypeError(".flowcraft.v1.AddSubGraph.edges: array expected");
                    message.edges = [];
                    for (let i = 0; i < object.edges.length; ++i) {
                        if (typeof object.edges[i] !== "object")
                            throw TypeError(".flowcraft.v1.AddSubGraph.edges: object expected");
                        message.edges[i] = $root.flowcraft.v1.Edge.fromObject(object.edges[i]);
                    }
                }
                return message;
            };

            /**
             * Creates a plain object from an AddSubGraph message. Also converts values to other types if specified.
             * @function toObject
             * @memberof flowcraft.v1.AddSubGraph
             * @static
             * @param {flowcraft.v1.AddSubGraph} message AddSubGraph
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            AddSubGraph.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                let object = {};
                if (options.arrays || options.defaults) {
                    object.nodes = [];
                    object.edges = [];
                }
                if (message.nodes && message.nodes.length) {
                    object.nodes = [];
                    for (let j = 0; j < message.nodes.length; ++j)
                        object.nodes[j] = $root.flowcraft.v1.Node.toObject(message.nodes[j], options);
                }
                if (message.edges && message.edges.length) {
                    object.edges = [];
                    for (let j = 0; j < message.edges.length; ++j)
                        object.edges[j] = $root.flowcraft.v1.Edge.toObject(message.edges[j], options);
                }
                return object;
            };

            /**
             * Converts this AddSubGraph to JSON.
             * @function toJSON
             * @memberof flowcraft.v1.AddSubGraph
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            AddSubGraph.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };

            /**
             * Gets the default type url for AddSubGraph
             * @function getTypeUrl
             * @memberof flowcraft.v1.AddSubGraph
             * @static
             * @param {string} [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns {string} The default type url
             */
            AddSubGraph.getTypeUrl = function getTypeUrl(typeUrlPrefix) {
                if (typeUrlPrefix === undefined) {
                    typeUrlPrefix = "type.googleapis.com";
                }
                return typeUrlPrefix + "/flowcraft.v1.AddSubGraph";
            };

            return AddSubGraph;
        })();

        v1.ClearGraph = (function() {

            /**
             * Properties of a ClearGraph.
             * @memberof flowcraft.v1
             * @interface IClearGraph
             */

            /**
             * Constructs a new ClearGraph.
             * @memberof flowcraft.v1
             * @classdesc Represents a ClearGraph.
             * @implements IClearGraph
             * @constructor
             * @param {flowcraft.v1.IClearGraph=} [properties] Properties to set
             */
            function ClearGraph(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * Creates a new ClearGraph instance using the specified properties.
             * @function create
             * @memberof flowcraft.v1.ClearGraph
             * @static
             * @param {flowcraft.v1.IClearGraph=} [properties] Properties to set
             * @returns {flowcraft.v1.ClearGraph} ClearGraph instance
             */
            ClearGraph.create = function create(properties) {
                return new ClearGraph(properties);
            };

            /**
             * Encodes the specified ClearGraph message. Does not implicitly {@link flowcraft.v1.ClearGraph.verify|verify} messages.
             * @function encode
             * @memberof flowcraft.v1.ClearGraph
             * @static
             * @param {flowcraft.v1.IClearGraph} message ClearGraph message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            ClearGraph.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                return writer;
            };

            /**
             * Encodes the specified ClearGraph message, length delimited. Does not implicitly {@link flowcraft.v1.ClearGraph.verify|verify} messages.
             * @function encodeDelimited
             * @memberof flowcraft.v1.ClearGraph
             * @static
             * @param {flowcraft.v1.IClearGraph} message ClearGraph message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            ClearGraph.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes a ClearGraph message from the specified reader or buffer.
             * @function decode
             * @memberof flowcraft.v1.ClearGraph
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {flowcraft.v1.ClearGraph} ClearGraph
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            ClearGraph.decode = function decode(reader, length, error) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.flowcraft.v1.ClearGraph();
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    if (tag === error)
                        break;
                    switch (tag >>> 3) {
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };

            /**
             * Decodes a ClearGraph message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof flowcraft.v1.ClearGraph
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {flowcraft.v1.ClearGraph} ClearGraph
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            ClearGraph.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies a ClearGraph message.
             * @function verify
             * @memberof flowcraft.v1.ClearGraph
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            ClearGraph.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                return null;
            };

            /**
             * Creates a ClearGraph message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof flowcraft.v1.ClearGraph
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {flowcraft.v1.ClearGraph} ClearGraph
             */
            ClearGraph.fromObject = function fromObject(object) {
                if (object instanceof $root.flowcraft.v1.ClearGraph)
                    return object;
                return new $root.flowcraft.v1.ClearGraph();
            };

            /**
             * Creates a plain object from a ClearGraph message. Also converts values to other types if specified.
             * @function toObject
             * @memberof flowcraft.v1.ClearGraph
             * @static
             * @param {flowcraft.v1.ClearGraph} message ClearGraph
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            ClearGraph.toObject = function toObject() {
                return {};
            };

            /**
             * Converts this ClearGraph to JSON.
             * @function toJSON
             * @memberof flowcraft.v1.ClearGraph
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            ClearGraph.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };

            /**
             * Gets the default type url for ClearGraph
             * @function getTypeUrl
             * @memberof flowcraft.v1.ClearGraph
             * @static
             * @param {string} [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns {string} The default type url
             */
            ClearGraph.getTypeUrl = function getTypeUrl(typeUrlPrefix) {
                if (typeUrlPrefix === undefined) {
                    typeUrlPrefix = "type.googleapis.com";
                }
                return typeUrlPrefix + "/flowcraft.v1.ClearGraph";
            };

            return ClearGraph;
        })();

        v1.TaskUpdate = (function() {

            /**
             * Properties of a TaskUpdate.
             * @memberof flowcraft.v1
             * @interface ITaskUpdate
             * @property {string|null} [taskId] TaskUpdate taskId
             * @property {flowcraft.v1.TaskStatus|null} [status] TaskUpdate status
             * @property {number|null} [progress] TaskUpdate progress
             * @property {string|null} [message] TaskUpdate message
             * @property {string|null} [resultJson] TaskUpdate resultJson
             */

            /**
             * Constructs a new TaskUpdate.
             * @memberof flowcraft.v1
             * @classdesc Represents a TaskUpdate.
             * @implements ITaskUpdate
             * @constructor
             * @param {flowcraft.v1.ITaskUpdate=} [properties] Properties to set
             */
            function TaskUpdate(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * TaskUpdate taskId.
             * @member {string} taskId
             * @memberof flowcraft.v1.TaskUpdate
             * @instance
             */
            TaskUpdate.prototype.taskId = "";

            /**
             * TaskUpdate status.
             * @member {flowcraft.v1.TaskStatus} status
             * @memberof flowcraft.v1.TaskUpdate
             * @instance
             */
            TaskUpdate.prototype.status = 0;

            /**
             * TaskUpdate progress.
             * @member {number} progress
             * @memberof flowcraft.v1.TaskUpdate
             * @instance
             */
            TaskUpdate.prototype.progress = 0;

            /**
             * TaskUpdate message.
             * @member {string} message
             * @memberof flowcraft.v1.TaskUpdate
             * @instance
             */
            TaskUpdate.prototype.message = "";

            /**
             * TaskUpdate resultJson.
             * @member {string} resultJson
             * @memberof flowcraft.v1.TaskUpdate
             * @instance
             */
            TaskUpdate.prototype.resultJson = "";

            /**
             * Creates a new TaskUpdate instance using the specified properties.
             * @function create
             * @memberof flowcraft.v1.TaskUpdate
             * @static
             * @param {flowcraft.v1.ITaskUpdate=} [properties] Properties to set
             * @returns {flowcraft.v1.TaskUpdate} TaskUpdate instance
             */
            TaskUpdate.create = function create(properties) {
                return new TaskUpdate(properties);
            };

            /**
             * Encodes the specified TaskUpdate message. Does not implicitly {@link flowcraft.v1.TaskUpdate.verify|verify} messages.
             * @function encode
             * @memberof flowcraft.v1.TaskUpdate
             * @static
             * @param {flowcraft.v1.ITaskUpdate} message TaskUpdate message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            TaskUpdate.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.taskId != null && Object.hasOwnProperty.call(message, "taskId"))
                    writer.uint32(/* id 1, wireType 2 =*/10).string(message.taskId);
                if (message.status != null && Object.hasOwnProperty.call(message, "status"))
                    writer.uint32(/* id 2, wireType 0 =*/16).int32(message.status);
                if (message.progress != null && Object.hasOwnProperty.call(message, "progress"))
                    writer.uint32(/* id 3, wireType 1 =*/25).double(message.progress);
                if (message.message != null && Object.hasOwnProperty.call(message, "message"))
                    writer.uint32(/* id 4, wireType 2 =*/34).string(message.message);
                if (message.resultJson != null && Object.hasOwnProperty.call(message, "resultJson"))
                    writer.uint32(/* id 5, wireType 2 =*/42).string(message.resultJson);
                return writer;
            };

            /**
             * Encodes the specified TaskUpdate message, length delimited. Does not implicitly {@link flowcraft.v1.TaskUpdate.verify|verify} messages.
             * @function encodeDelimited
             * @memberof flowcraft.v1.TaskUpdate
             * @static
             * @param {flowcraft.v1.ITaskUpdate} message TaskUpdate message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            TaskUpdate.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes a TaskUpdate message from the specified reader or buffer.
             * @function decode
             * @memberof flowcraft.v1.TaskUpdate
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {flowcraft.v1.TaskUpdate} TaskUpdate
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            TaskUpdate.decode = function decode(reader, length, error) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.flowcraft.v1.TaskUpdate();
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    if (tag === error)
                        break;
                    switch (tag >>> 3) {
                    case 1: {
                            message.taskId = reader.string();
                            break;
                        }
                    case 2: {
                            message.status = reader.int32();
                            break;
                        }
                    case 3: {
                            message.progress = reader.double();
                            break;
                        }
                    case 4: {
                            message.message = reader.string();
                            break;
                        }
                    case 5: {
                            message.resultJson = reader.string();
                            break;
                        }
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };

            /**
             * Decodes a TaskUpdate message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof flowcraft.v1.TaskUpdate
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {flowcraft.v1.TaskUpdate} TaskUpdate
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            TaskUpdate.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies a TaskUpdate message.
             * @function verify
             * @memberof flowcraft.v1.TaskUpdate
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            TaskUpdate.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.taskId != null && message.hasOwnProperty("taskId"))
                    if (!$util.isString(message.taskId))
                        return "taskId: string expected";
                if (message.status != null && message.hasOwnProperty("status"))
                    switch (message.status) {
                    default:
                        return "status: enum value expected";
                    case 0:
                    case 1:
                    case 2:
                    case 3:
                    case 4:
                        break;
                    }
                if (message.progress != null && message.hasOwnProperty("progress"))
                    if (typeof message.progress !== "number")
                        return "progress: number expected";
                if (message.message != null && message.hasOwnProperty("message"))
                    if (!$util.isString(message.message))
                        return "message: string expected";
                if (message.resultJson != null && message.hasOwnProperty("resultJson"))
                    if (!$util.isString(message.resultJson))
                        return "resultJson: string expected";
                return null;
            };

            /**
             * Creates a TaskUpdate message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof flowcraft.v1.TaskUpdate
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {flowcraft.v1.TaskUpdate} TaskUpdate
             */
            TaskUpdate.fromObject = function fromObject(object) {
                if (object instanceof $root.flowcraft.v1.TaskUpdate)
                    return object;
                let message = new $root.flowcraft.v1.TaskUpdate();
                if (object.taskId != null)
                    message.taskId = String(object.taskId);
                switch (object.status) {
                default:
                    if (typeof object.status === "number") {
                        message.status = object.status;
                        break;
                    }
                    break;
                case "TASK_PENDING":
                case 0:
                    message.status = 0;
                    break;
                case "TASK_PROCESSING":
                case 1:
                    message.status = 1;
                    break;
                case "TASK_COMPLETED":
                case 2:
                    message.status = 2;
                    break;
                case "TASK_FAILED":
                case 3:
                    message.status = 3;
                    break;
                case "TASK_CANCELLED":
                case 4:
                    message.status = 4;
                    break;
                }
                if (object.progress != null)
                    message.progress = Number(object.progress);
                if (object.message != null)
                    message.message = String(object.message);
                if (object.resultJson != null)
                    message.resultJson = String(object.resultJson);
                return message;
            };

            /**
             * Creates a plain object from a TaskUpdate message. Also converts values to other types if specified.
             * @function toObject
             * @memberof flowcraft.v1.TaskUpdate
             * @static
             * @param {flowcraft.v1.TaskUpdate} message TaskUpdate
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            TaskUpdate.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                let object = {};
                if (options.defaults) {
                    object.taskId = "";
                    object.status = options.enums === String ? "TASK_PENDING" : 0;
                    object.progress = 0;
                    object.message = "";
                    object.resultJson = "";
                }
                if (message.taskId != null && message.hasOwnProperty("taskId"))
                    object.taskId = message.taskId;
                if (message.status != null && message.hasOwnProperty("status"))
                    object.status = options.enums === String ? $root.flowcraft.v1.TaskStatus[message.status] === undefined ? message.status : $root.flowcraft.v1.TaskStatus[message.status] : message.status;
                if (message.progress != null && message.hasOwnProperty("progress"))
                    object.progress = options.json && !isFinite(message.progress) ? String(message.progress) : message.progress;
                if (message.message != null && message.hasOwnProperty("message"))
                    object.message = message.message;
                if (message.resultJson != null && message.hasOwnProperty("resultJson"))
                    object.resultJson = message.resultJson;
                return object;
            };

            /**
             * Converts this TaskUpdate to JSON.
             * @function toJSON
             * @memberof flowcraft.v1.TaskUpdate
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            TaskUpdate.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };

            /**
             * Gets the default type url for TaskUpdate
             * @function getTypeUrl
             * @memberof flowcraft.v1.TaskUpdate
             * @static
             * @param {string} [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns {string} The default type url
             */
            TaskUpdate.getTypeUrl = function getTypeUrl(typeUrlPrefix) {
                if (typeUrlPrefix === undefined) {
                    typeUrlPrefix = "type.googleapis.com";
                }
                return typeUrlPrefix + "/flowcraft.v1.TaskUpdate";
            };

            return TaskUpdate;
        })();

        /**
         * TaskStatus enum.
         * @name flowcraft.v1.TaskStatus
         * @enum {number}
         * @property {number} TASK_PENDING=0 TASK_PENDING value
         * @property {number} TASK_PROCESSING=1 TASK_PROCESSING value
         * @property {number} TASK_COMPLETED=2 TASK_COMPLETED value
         * @property {number} TASK_FAILED=3 TASK_FAILED value
         * @property {number} TASK_CANCELLED=4 TASK_CANCELLED value
         */
        v1.TaskStatus = (function() {
            const valuesById = {}, values = Object.create(valuesById);
            values[valuesById[0] = "TASK_PENDING"] = 0;
            values[valuesById[1] = "TASK_PROCESSING"] = 1;
            values[valuesById[2] = "TASK_COMPLETED"] = 2;
            values[valuesById[3] = "TASK_FAILED"] = 3;
            values[valuesById[4] = "TASK_CANCELLED"] = 4;
            return values;
        })();

        v1.TaskCancelRequest = (function() {

            /**
             * Properties of a TaskCancelRequest.
             * @memberof flowcraft.v1
             * @interface ITaskCancelRequest
             * @property {string|null} [taskId] TaskCancelRequest taskId
             */

            /**
             * Constructs a new TaskCancelRequest.
             * @memberof flowcraft.v1
             * @classdesc Represents a TaskCancelRequest.
             * @implements ITaskCancelRequest
             * @constructor
             * @param {flowcraft.v1.ITaskCancelRequest=} [properties] Properties to set
             */
            function TaskCancelRequest(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * TaskCancelRequest taskId.
             * @member {string} taskId
             * @memberof flowcraft.v1.TaskCancelRequest
             * @instance
             */
            TaskCancelRequest.prototype.taskId = "";

            /**
             * Creates a new TaskCancelRequest instance using the specified properties.
             * @function create
             * @memberof flowcraft.v1.TaskCancelRequest
             * @static
             * @param {flowcraft.v1.ITaskCancelRequest=} [properties] Properties to set
             * @returns {flowcraft.v1.TaskCancelRequest} TaskCancelRequest instance
             */
            TaskCancelRequest.create = function create(properties) {
                return new TaskCancelRequest(properties);
            };

            /**
             * Encodes the specified TaskCancelRequest message. Does not implicitly {@link flowcraft.v1.TaskCancelRequest.verify|verify} messages.
             * @function encode
             * @memberof flowcraft.v1.TaskCancelRequest
             * @static
             * @param {flowcraft.v1.ITaskCancelRequest} message TaskCancelRequest message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            TaskCancelRequest.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.taskId != null && Object.hasOwnProperty.call(message, "taskId"))
                    writer.uint32(/* id 1, wireType 2 =*/10).string(message.taskId);
                return writer;
            };

            /**
             * Encodes the specified TaskCancelRequest message, length delimited. Does not implicitly {@link flowcraft.v1.TaskCancelRequest.verify|verify} messages.
             * @function encodeDelimited
             * @memberof flowcraft.v1.TaskCancelRequest
             * @static
             * @param {flowcraft.v1.ITaskCancelRequest} message TaskCancelRequest message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            TaskCancelRequest.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes a TaskCancelRequest message from the specified reader or buffer.
             * @function decode
             * @memberof flowcraft.v1.TaskCancelRequest
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {flowcraft.v1.TaskCancelRequest} TaskCancelRequest
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            TaskCancelRequest.decode = function decode(reader, length, error) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.flowcraft.v1.TaskCancelRequest();
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    if (tag === error)
                        break;
                    switch (tag >>> 3) {
                    case 1: {
                            message.taskId = reader.string();
                            break;
                        }
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };

            /**
             * Decodes a TaskCancelRequest message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof flowcraft.v1.TaskCancelRequest
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {flowcraft.v1.TaskCancelRequest} TaskCancelRequest
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            TaskCancelRequest.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies a TaskCancelRequest message.
             * @function verify
             * @memberof flowcraft.v1.TaskCancelRequest
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            TaskCancelRequest.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.taskId != null && message.hasOwnProperty("taskId"))
                    if (!$util.isString(message.taskId))
                        return "taskId: string expected";
                return null;
            };

            /**
             * Creates a TaskCancelRequest message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof flowcraft.v1.TaskCancelRequest
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {flowcraft.v1.TaskCancelRequest} TaskCancelRequest
             */
            TaskCancelRequest.fromObject = function fromObject(object) {
                if (object instanceof $root.flowcraft.v1.TaskCancelRequest)
                    return object;
                let message = new $root.flowcraft.v1.TaskCancelRequest();
                if (object.taskId != null)
                    message.taskId = String(object.taskId);
                return message;
            };

            /**
             * Creates a plain object from a TaskCancelRequest message. Also converts values to other types if specified.
             * @function toObject
             * @memberof flowcraft.v1.TaskCancelRequest
             * @static
             * @param {flowcraft.v1.TaskCancelRequest} message TaskCancelRequest
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            TaskCancelRequest.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                let object = {};
                if (options.defaults)
                    object.taskId = "";
                if (message.taskId != null && message.hasOwnProperty("taskId"))
                    object.taskId = message.taskId;
                return object;
            };

            /**
             * Converts this TaskCancelRequest to JSON.
             * @function toJSON
             * @memberof flowcraft.v1.TaskCancelRequest
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            TaskCancelRequest.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };

            /**
             * Gets the default type url for TaskCancelRequest
             * @function getTypeUrl
             * @memberof flowcraft.v1.TaskCancelRequest
             * @static
             * @param {string} [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns {string} The default type url
             */
            TaskCancelRequest.getTypeUrl = function getTypeUrl(typeUrlPrefix) {
                if (typeUrlPrefix === undefined) {
                    typeUrlPrefix = "type.googleapis.com";
                }
                return typeUrlPrefix + "/flowcraft.v1.TaskCancelRequest";
            };

            return TaskCancelRequest;
        })();

        v1.Node = (function() {

            /**
             * Properties of a Node.
             * @memberof flowcraft.v1
             * @interface INode
             * @property {string|null} [id] Node id
             * @property {string|null} [type] Node type
             * @property {flowcraft.v1.IPosition|null} [position] Node position
             * @property {flowcraft.v1.INodeData|null} [data] Node data
             * @property {number|null} [width] Node width
             * @property {number|null} [height] Node height
             * @property {boolean|null} [selected] Node selected
             * @property {string|null} [parentId] Node parentId
             */

            /**
             * Constructs a new Node.
             * @memberof flowcraft.v1
             * @classdesc Represents a Node.
             * @implements INode
             * @constructor
             * @param {flowcraft.v1.INode=} [properties] Properties to set
             */
            function Node(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * Node id.
             * @member {string} id
             * @memberof flowcraft.v1.Node
             * @instance
             */
            Node.prototype.id = "";

            /**
             * Node type.
             * @member {string} type
             * @memberof flowcraft.v1.Node
             * @instance
             */
            Node.prototype.type = "";

            /**
             * Node position.
             * @member {flowcraft.v1.IPosition|null|undefined} position
             * @memberof flowcraft.v1.Node
             * @instance
             */
            Node.prototype.position = null;

            /**
             * Node data.
             * @member {flowcraft.v1.INodeData|null|undefined} data
             * @memberof flowcraft.v1.Node
             * @instance
             */
            Node.prototype.data = null;

            /**
             * Node width.
             * @member {number} width
             * @memberof flowcraft.v1.Node
             * @instance
             */
            Node.prototype.width = 0;

            /**
             * Node height.
             * @member {number} height
             * @memberof flowcraft.v1.Node
             * @instance
             */
            Node.prototype.height = 0;

            /**
             * Node selected.
             * @member {boolean} selected
             * @memberof flowcraft.v1.Node
             * @instance
             */
            Node.prototype.selected = false;

            /**
             * Node parentId.
             * @member {string} parentId
             * @memberof flowcraft.v1.Node
             * @instance
             */
            Node.prototype.parentId = "";

            /**
             * Creates a new Node instance using the specified properties.
             * @function create
             * @memberof flowcraft.v1.Node
             * @static
             * @param {flowcraft.v1.INode=} [properties] Properties to set
             * @returns {flowcraft.v1.Node} Node instance
             */
            Node.create = function create(properties) {
                return new Node(properties);
            };

            /**
             * Encodes the specified Node message. Does not implicitly {@link flowcraft.v1.Node.verify|verify} messages.
             * @function encode
             * @memberof flowcraft.v1.Node
             * @static
             * @param {flowcraft.v1.INode} message Node message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            Node.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.id != null && Object.hasOwnProperty.call(message, "id"))
                    writer.uint32(/* id 1, wireType 2 =*/10).string(message.id);
                if (message.type != null && Object.hasOwnProperty.call(message, "type"))
                    writer.uint32(/* id 2, wireType 2 =*/18).string(message.type);
                if (message.position != null && Object.hasOwnProperty.call(message, "position"))
                    $root.flowcraft.v1.Position.encode(message.position, writer.uint32(/* id 3, wireType 2 =*/26).fork()).ldelim();
                if (message.data != null && Object.hasOwnProperty.call(message, "data"))
                    $root.flowcraft.v1.NodeData.encode(message.data, writer.uint32(/* id 4, wireType 2 =*/34).fork()).ldelim();
                if (message.width != null && Object.hasOwnProperty.call(message, "width"))
                    writer.uint32(/* id 5, wireType 1 =*/41).double(message.width);
                if (message.height != null && Object.hasOwnProperty.call(message, "height"))
                    writer.uint32(/* id 6, wireType 1 =*/49).double(message.height);
                if (message.selected != null && Object.hasOwnProperty.call(message, "selected"))
                    writer.uint32(/* id 7, wireType 0 =*/56).bool(message.selected);
                if (message.parentId != null && Object.hasOwnProperty.call(message, "parentId"))
                    writer.uint32(/* id 8, wireType 2 =*/66).string(message.parentId);
                return writer;
            };

            /**
             * Encodes the specified Node message, length delimited. Does not implicitly {@link flowcraft.v1.Node.verify|verify} messages.
             * @function encodeDelimited
             * @memberof flowcraft.v1.Node
             * @static
             * @param {flowcraft.v1.INode} message Node message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            Node.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes a Node message from the specified reader or buffer.
             * @function decode
             * @memberof flowcraft.v1.Node
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {flowcraft.v1.Node} Node
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            Node.decode = function decode(reader, length, error) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.flowcraft.v1.Node();
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    if (tag === error)
                        break;
                    switch (tag >>> 3) {
                    case 1: {
                            message.id = reader.string();
                            break;
                        }
                    case 2: {
                            message.type = reader.string();
                            break;
                        }
                    case 3: {
                            message.position = $root.flowcraft.v1.Position.decode(reader, reader.uint32());
                            break;
                        }
                    case 4: {
                            message.data = $root.flowcraft.v1.NodeData.decode(reader, reader.uint32());
                            break;
                        }
                    case 5: {
                            message.width = reader.double();
                            break;
                        }
                    case 6: {
                            message.height = reader.double();
                            break;
                        }
                    case 7: {
                            message.selected = reader.bool();
                            break;
                        }
                    case 8: {
                            message.parentId = reader.string();
                            break;
                        }
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };

            /**
             * Decodes a Node message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof flowcraft.v1.Node
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {flowcraft.v1.Node} Node
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            Node.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies a Node message.
             * @function verify
             * @memberof flowcraft.v1.Node
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            Node.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.id != null && message.hasOwnProperty("id"))
                    if (!$util.isString(message.id))
                        return "id: string expected";
                if (message.type != null && message.hasOwnProperty("type"))
                    if (!$util.isString(message.type))
                        return "type: string expected";
                if (message.position != null && message.hasOwnProperty("position")) {
                    let error = $root.flowcraft.v1.Position.verify(message.position);
                    if (error)
                        return "position." + error;
                }
                if (message.data != null && message.hasOwnProperty("data")) {
                    let error = $root.flowcraft.v1.NodeData.verify(message.data);
                    if (error)
                        return "data." + error;
                }
                if (message.width != null && message.hasOwnProperty("width"))
                    if (typeof message.width !== "number")
                        return "width: number expected";
                if (message.height != null && message.hasOwnProperty("height"))
                    if (typeof message.height !== "number")
                        return "height: number expected";
                if (message.selected != null && message.hasOwnProperty("selected"))
                    if (typeof message.selected !== "boolean")
                        return "selected: boolean expected";
                if (message.parentId != null && message.hasOwnProperty("parentId"))
                    if (!$util.isString(message.parentId))
                        return "parentId: string expected";
                return null;
            };

            /**
             * Creates a Node message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof flowcraft.v1.Node
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {flowcraft.v1.Node} Node
             */
            Node.fromObject = function fromObject(object) {
                if (object instanceof $root.flowcraft.v1.Node)
                    return object;
                let message = new $root.flowcraft.v1.Node();
                if (object.id != null)
                    message.id = String(object.id);
                if (object.type != null)
                    message.type = String(object.type);
                if (object.position != null) {
                    if (typeof object.position !== "object")
                        throw TypeError(".flowcraft.v1.Node.position: object expected");
                    message.position = $root.flowcraft.v1.Position.fromObject(object.position);
                }
                if (object.data != null) {
                    if (typeof object.data !== "object")
                        throw TypeError(".flowcraft.v1.Node.data: object expected");
                    message.data = $root.flowcraft.v1.NodeData.fromObject(object.data);
                }
                if (object.width != null)
                    message.width = Number(object.width);
                if (object.height != null)
                    message.height = Number(object.height);
                if (object.selected != null)
                    message.selected = Boolean(object.selected);
                if (object.parentId != null)
                    message.parentId = String(object.parentId);
                return message;
            };

            /**
             * Creates a plain object from a Node message. Also converts values to other types if specified.
             * @function toObject
             * @memberof flowcraft.v1.Node
             * @static
             * @param {flowcraft.v1.Node} message Node
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            Node.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                let object = {};
                if (options.defaults) {
                    object.id = "";
                    object.type = "";
                    object.position = null;
                    object.data = null;
                    object.width = 0;
                    object.height = 0;
                    object.selected = false;
                    object.parentId = "";
                }
                if (message.id != null && message.hasOwnProperty("id"))
                    object.id = message.id;
                if (message.type != null && message.hasOwnProperty("type"))
                    object.type = message.type;
                if (message.position != null && message.hasOwnProperty("position"))
                    object.position = $root.flowcraft.v1.Position.toObject(message.position, options);
                if (message.data != null && message.hasOwnProperty("data"))
                    object.data = $root.flowcraft.v1.NodeData.toObject(message.data, options);
                if (message.width != null && message.hasOwnProperty("width"))
                    object.width = options.json && !isFinite(message.width) ? String(message.width) : message.width;
                if (message.height != null && message.hasOwnProperty("height"))
                    object.height = options.json && !isFinite(message.height) ? String(message.height) : message.height;
                if (message.selected != null && message.hasOwnProperty("selected"))
                    object.selected = message.selected;
                if (message.parentId != null && message.hasOwnProperty("parentId"))
                    object.parentId = message.parentId;
                return object;
            };

            /**
             * Converts this Node to JSON.
             * @function toJSON
             * @memberof flowcraft.v1.Node
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            Node.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };

            /**
             * Gets the default type url for Node
             * @function getTypeUrl
             * @memberof flowcraft.v1.Node
             * @static
             * @param {string} [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns {string} The default type url
             */
            Node.getTypeUrl = function getTypeUrl(typeUrlPrefix) {
                if (typeUrlPrefix === undefined) {
                    typeUrlPrefix = "type.googleapis.com";
                }
                return typeUrlPrefix + "/flowcraft.v1.Node";
            };

            return Node;
        })();

        v1.NodeData = (function() {

            /**
             * Properties of a NodeData.
             * @memberof flowcraft.v1
             * @interface INodeData
             * @property {string|null} [label] NodeData label
             * @property {Array.<flowcraft.v1.RenderMode>|null} [availableModes] NodeData availableModes
             * @property {flowcraft.v1.RenderMode|null} [activeMode] NodeData activeMode
             * @property {flowcraft.v1.IMediaContent|null} [media] NodeData media
             * @property {Array.<flowcraft.v1.IWidget>|null} [widgets] NodeData widgets
             * @property {Array.<flowcraft.v1.IPort>|null} [inputPorts] NodeData inputPorts
             * @property {Array.<flowcraft.v1.IPort>|null} [outputPorts] NodeData outputPorts
             * @property {Object.<string,string>|null} [metadata] NodeData metadata
             */

            /**
             * Constructs a new NodeData.
             * @memberof flowcraft.v1
             * @classdesc Represents a NodeData.
             * @implements INodeData
             * @constructor
             * @param {flowcraft.v1.INodeData=} [properties] Properties to set
             */
            function NodeData(properties) {
                this.availableModes = [];
                this.widgets = [];
                this.inputPorts = [];
                this.outputPorts = [];
                this.metadata = {};
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * NodeData label.
             * @member {string} label
             * @memberof flowcraft.v1.NodeData
             * @instance
             */
            NodeData.prototype.label = "";

            /**
             * NodeData availableModes.
             * @member {Array.<flowcraft.v1.RenderMode>} availableModes
             * @memberof flowcraft.v1.NodeData
             * @instance
             */
            NodeData.prototype.availableModes = $util.emptyArray;

            /**
             * NodeData activeMode.
             * @member {flowcraft.v1.RenderMode} activeMode
             * @memberof flowcraft.v1.NodeData
             * @instance
             */
            NodeData.prototype.activeMode = 0;

            /**
             * NodeData media.
             * @member {flowcraft.v1.IMediaContent|null|undefined} media
             * @memberof flowcraft.v1.NodeData
             * @instance
             */
            NodeData.prototype.media = null;

            /**
             * NodeData widgets.
             * @member {Array.<flowcraft.v1.IWidget>} widgets
             * @memberof flowcraft.v1.NodeData
             * @instance
             */
            NodeData.prototype.widgets = $util.emptyArray;

            /**
             * NodeData inputPorts.
             * @member {Array.<flowcraft.v1.IPort>} inputPorts
             * @memberof flowcraft.v1.NodeData
             * @instance
             */
            NodeData.prototype.inputPorts = $util.emptyArray;

            /**
             * NodeData outputPorts.
             * @member {Array.<flowcraft.v1.IPort>} outputPorts
             * @memberof flowcraft.v1.NodeData
             * @instance
             */
            NodeData.prototype.outputPorts = $util.emptyArray;

            /**
             * NodeData metadata.
             * @member {Object.<string,string>} metadata
             * @memberof flowcraft.v1.NodeData
             * @instance
             */
            NodeData.prototype.metadata = $util.emptyObject;

            /**
             * Creates a new NodeData instance using the specified properties.
             * @function create
             * @memberof flowcraft.v1.NodeData
             * @static
             * @param {flowcraft.v1.INodeData=} [properties] Properties to set
             * @returns {flowcraft.v1.NodeData} NodeData instance
             */
            NodeData.create = function create(properties) {
                return new NodeData(properties);
            };

            /**
             * Encodes the specified NodeData message. Does not implicitly {@link flowcraft.v1.NodeData.verify|verify} messages.
             * @function encode
             * @memberof flowcraft.v1.NodeData
             * @static
             * @param {flowcraft.v1.INodeData} message NodeData message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            NodeData.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.label != null && Object.hasOwnProperty.call(message, "label"))
                    writer.uint32(/* id 1, wireType 2 =*/10).string(message.label);
                if (message.availableModes != null && message.availableModes.length) {
                    writer.uint32(/* id 2, wireType 2 =*/18).fork();
                    for (let i = 0; i < message.availableModes.length; ++i)
                        writer.int32(message.availableModes[i]);
                    writer.ldelim();
                }
                if (message.activeMode != null && Object.hasOwnProperty.call(message, "activeMode"))
                    writer.uint32(/* id 3, wireType 0 =*/24).int32(message.activeMode);
                if (message.media != null && Object.hasOwnProperty.call(message, "media"))
                    $root.flowcraft.v1.MediaContent.encode(message.media, writer.uint32(/* id 4, wireType 2 =*/34).fork()).ldelim();
                if (message.widgets != null && message.widgets.length)
                    for (let i = 0; i < message.widgets.length; ++i)
                        $root.flowcraft.v1.Widget.encode(message.widgets[i], writer.uint32(/* id 5, wireType 2 =*/42).fork()).ldelim();
                if (message.inputPorts != null && message.inputPorts.length)
                    for (let i = 0; i < message.inputPorts.length; ++i)
                        $root.flowcraft.v1.Port.encode(message.inputPorts[i], writer.uint32(/* id 6, wireType 2 =*/50).fork()).ldelim();
                if (message.outputPorts != null && message.outputPorts.length)
                    for (let i = 0; i < message.outputPorts.length; ++i)
                        $root.flowcraft.v1.Port.encode(message.outputPorts[i], writer.uint32(/* id 7, wireType 2 =*/58).fork()).ldelim();
                if (message.metadata != null && Object.hasOwnProperty.call(message, "metadata"))
                    for (let keys = Object.keys(message.metadata), i = 0; i < keys.length; ++i)
                        writer.uint32(/* id 8, wireType 2 =*/66).fork().uint32(/* id 1, wireType 2 =*/10).string(keys[i]).uint32(/* id 2, wireType 2 =*/18).string(message.metadata[keys[i]]).ldelim();
                return writer;
            };

            /**
             * Encodes the specified NodeData message, length delimited. Does not implicitly {@link flowcraft.v1.NodeData.verify|verify} messages.
             * @function encodeDelimited
             * @memberof flowcraft.v1.NodeData
             * @static
             * @param {flowcraft.v1.INodeData} message NodeData message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            NodeData.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes a NodeData message from the specified reader or buffer.
             * @function decode
             * @memberof flowcraft.v1.NodeData
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {flowcraft.v1.NodeData} NodeData
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            NodeData.decode = function decode(reader, length, error) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.flowcraft.v1.NodeData(), key, value;
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    if (tag === error)
                        break;
                    switch (tag >>> 3) {
                    case 1: {
                            message.label = reader.string();
                            break;
                        }
                    case 2: {
                            if (!(message.availableModes && message.availableModes.length))
                                message.availableModes = [];
                            if ((tag & 7) === 2) {
                                let end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.availableModes.push(reader.int32());
                            } else
                                message.availableModes.push(reader.int32());
                            break;
                        }
                    case 3: {
                            message.activeMode = reader.int32();
                            break;
                        }
                    case 4: {
                            message.media = $root.flowcraft.v1.MediaContent.decode(reader, reader.uint32());
                            break;
                        }
                    case 5: {
                            if (!(message.widgets && message.widgets.length))
                                message.widgets = [];
                            message.widgets.push($root.flowcraft.v1.Widget.decode(reader, reader.uint32()));
                            break;
                        }
                    case 6: {
                            if (!(message.inputPorts && message.inputPorts.length))
                                message.inputPorts = [];
                            message.inputPorts.push($root.flowcraft.v1.Port.decode(reader, reader.uint32()));
                            break;
                        }
                    case 7: {
                            if (!(message.outputPorts && message.outputPorts.length))
                                message.outputPorts = [];
                            message.outputPorts.push($root.flowcraft.v1.Port.decode(reader, reader.uint32()));
                            break;
                        }
                    case 8: {
                            if (message.metadata === $util.emptyObject)
                                message.metadata = {};
                            let end2 = reader.uint32() + reader.pos;
                            key = "";
                            value = "";
                            while (reader.pos < end2) {
                                let tag2 = reader.uint32();
                                switch (tag2 >>> 3) {
                                case 1:
                                    key = reader.string();
                                    break;
                                case 2:
                                    value = reader.string();
                                    break;
                                default:
                                    reader.skipType(tag2 & 7);
                                    break;
                                }
                            }
                            message.metadata[key] = value;
                            break;
                        }
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };

            /**
             * Decodes a NodeData message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof flowcraft.v1.NodeData
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {flowcraft.v1.NodeData} NodeData
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            NodeData.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies a NodeData message.
             * @function verify
             * @memberof flowcraft.v1.NodeData
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            NodeData.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.label != null && message.hasOwnProperty("label"))
                    if (!$util.isString(message.label))
                        return "label: string expected";
                if (message.availableModes != null && message.hasOwnProperty("availableModes")) {
                    if (!Array.isArray(message.availableModes))
                        return "availableModes: array expected";
                    for (let i = 0; i < message.availableModes.length; ++i)
                        switch (message.availableModes[i]) {
                        default:
                            return "availableModes: enum value[] expected";
                        case 0:
                        case 1:
                        case 2:
                        case 3:
                            break;
                        }
                }
                if (message.activeMode != null && message.hasOwnProperty("activeMode"))
                    switch (message.activeMode) {
                    default:
                        return "activeMode: enum value expected";
                    case 0:
                    case 1:
                    case 2:
                    case 3:
                        break;
                    }
                if (message.media != null && message.hasOwnProperty("media")) {
                    let error = $root.flowcraft.v1.MediaContent.verify(message.media);
                    if (error)
                        return "media." + error;
                }
                if (message.widgets != null && message.hasOwnProperty("widgets")) {
                    if (!Array.isArray(message.widgets))
                        return "widgets: array expected";
                    for (let i = 0; i < message.widgets.length; ++i) {
                        let error = $root.flowcraft.v1.Widget.verify(message.widgets[i]);
                        if (error)
                            return "widgets." + error;
                    }
                }
                if (message.inputPorts != null && message.hasOwnProperty("inputPorts")) {
                    if (!Array.isArray(message.inputPorts))
                        return "inputPorts: array expected";
                    for (let i = 0; i < message.inputPorts.length; ++i) {
                        let error = $root.flowcraft.v1.Port.verify(message.inputPorts[i]);
                        if (error)
                            return "inputPorts." + error;
                    }
                }
                if (message.outputPorts != null && message.hasOwnProperty("outputPorts")) {
                    if (!Array.isArray(message.outputPorts))
                        return "outputPorts: array expected";
                    for (let i = 0; i < message.outputPorts.length; ++i) {
                        let error = $root.flowcraft.v1.Port.verify(message.outputPorts[i]);
                        if (error)
                            return "outputPorts." + error;
                    }
                }
                if (message.metadata != null && message.hasOwnProperty("metadata")) {
                    if (!$util.isObject(message.metadata))
                        return "metadata: object expected";
                    let key = Object.keys(message.metadata);
                    for (let i = 0; i < key.length; ++i)
                        if (!$util.isString(message.metadata[key[i]]))
                            return "metadata: string{k:string} expected";
                }
                return null;
            };

            /**
             * Creates a NodeData message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof flowcraft.v1.NodeData
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {flowcraft.v1.NodeData} NodeData
             */
            NodeData.fromObject = function fromObject(object) {
                if (object instanceof $root.flowcraft.v1.NodeData)
                    return object;
                let message = new $root.flowcraft.v1.NodeData();
                if (object.label != null)
                    message.label = String(object.label);
                if (object.availableModes) {
                    if (!Array.isArray(object.availableModes))
                        throw TypeError(".flowcraft.v1.NodeData.availableModes: array expected");
                    message.availableModes = [];
                    for (let i = 0; i < object.availableModes.length; ++i)
                        switch (object.availableModes[i]) {
                        default:
                            if (typeof object.availableModes[i] === "number") {
                                message.availableModes[i] = object.availableModes[i];
                                break;
                            }
                        case "MODE_UNSPECIFIED":
                        case 0:
                            message.availableModes[i] = 0;
                            break;
                        case "MODE_MEDIA":
                        case 1:
                            message.availableModes[i] = 1;
                            break;
                        case "MODE_WIDGETS":
                        case 2:
                            message.availableModes[i] = 2;
                            break;
                        case "MODE_MARKDOWN":
                        case 3:
                            message.availableModes[i] = 3;
                            break;
                        }
                }
                switch (object.activeMode) {
                default:
                    if (typeof object.activeMode === "number") {
                        message.activeMode = object.activeMode;
                        break;
                    }
                    break;
                case "MODE_UNSPECIFIED":
                case 0:
                    message.activeMode = 0;
                    break;
                case "MODE_MEDIA":
                case 1:
                    message.activeMode = 1;
                    break;
                case "MODE_WIDGETS":
                case 2:
                    message.activeMode = 2;
                    break;
                case "MODE_MARKDOWN":
                case 3:
                    message.activeMode = 3;
                    break;
                }
                if (object.media != null) {
                    if (typeof object.media !== "object")
                        throw TypeError(".flowcraft.v1.NodeData.media: object expected");
                    message.media = $root.flowcraft.v1.MediaContent.fromObject(object.media);
                }
                if (object.widgets) {
                    if (!Array.isArray(object.widgets))
                        throw TypeError(".flowcraft.v1.NodeData.widgets: array expected");
                    message.widgets = [];
                    for (let i = 0; i < object.widgets.length; ++i) {
                        if (typeof object.widgets[i] !== "object")
                            throw TypeError(".flowcraft.v1.NodeData.widgets: object expected");
                        message.widgets[i] = $root.flowcraft.v1.Widget.fromObject(object.widgets[i]);
                    }
                }
                if (object.inputPorts) {
                    if (!Array.isArray(object.inputPorts))
                        throw TypeError(".flowcraft.v1.NodeData.inputPorts: array expected");
                    message.inputPorts = [];
                    for (let i = 0; i < object.inputPorts.length; ++i) {
                        if (typeof object.inputPorts[i] !== "object")
                            throw TypeError(".flowcraft.v1.NodeData.inputPorts: object expected");
                        message.inputPorts[i] = $root.flowcraft.v1.Port.fromObject(object.inputPorts[i]);
                    }
                }
                if (object.outputPorts) {
                    if (!Array.isArray(object.outputPorts))
                        throw TypeError(".flowcraft.v1.NodeData.outputPorts: array expected");
                    message.outputPorts = [];
                    for (let i = 0; i < object.outputPorts.length; ++i) {
                        if (typeof object.outputPorts[i] !== "object")
                            throw TypeError(".flowcraft.v1.NodeData.outputPorts: object expected");
                        message.outputPorts[i] = $root.flowcraft.v1.Port.fromObject(object.outputPorts[i]);
                    }
                }
                if (object.metadata) {
                    if (typeof object.metadata !== "object")
                        throw TypeError(".flowcraft.v1.NodeData.metadata: object expected");
                    message.metadata = {};
                    for (let keys = Object.keys(object.metadata), i = 0; i < keys.length; ++i)
                        message.metadata[keys[i]] = String(object.metadata[keys[i]]);
                }
                return message;
            };

            /**
             * Creates a plain object from a NodeData message. Also converts values to other types if specified.
             * @function toObject
             * @memberof flowcraft.v1.NodeData
             * @static
             * @param {flowcraft.v1.NodeData} message NodeData
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            NodeData.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                let object = {};
                if (options.arrays || options.defaults) {
                    object.availableModes = [];
                    object.widgets = [];
                    object.inputPorts = [];
                    object.outputPorts = [];
                }
                if (options.objects || options.defaults)
                    object.metadata = {};
                if (options.defaults) {
                    object.label = "";
                    object.activeMode = options.enums === String ? "MODE_UNSPECIFIED" : 0;
                    object.media = null;
                }
                if (message.label != null && message.hasOwnProperty("label"))
                    object.label = message.label;
                if (message.availableModes && message.availableModes.length) {
                    object.availableModes = [];
                    for (let j = 0; j < message.availableModes.length; ++j)
                        object.availableModes[j] = options.enums === String ? $root.flowcraft.v1.RenderMode[message.availableModes[j]] === undefined ? message.availableModes[j] : $root.flowcraft.v1.RenderMode[message.availableModes[j]] : message.availableModes[j];
                }
                if (message.activeMode != null && message.hasOwnProperty("activeMode"))
                    object.activeMode = options.enums === String ? $root.flowcraft.v1.RenderMode[message.activeMode] === undefined ? message.activeMode : $root.flowcraft.v1.RenderMode[message.activeMode] : message.activeMode;
                if (message.media != null && message.hasOwnProperty("media"))
                    object.media = $root.flowcraft.v1.MediaContent.toObject(message.media, options);
                if (message.widgets && message.widgets.length) {
                    object.widgets = [];
                    for (let j = 0; j < message.widgets.length; ++j)
                        object.widgets[j] = $root.flowcraft.v1.Widget.toObject(message.widgets[j], options);
                }
                if (message.inputPorts && message.inputPorts.length) {
                    object.inputPorts = [];
                    for (let j = 0; j < message.inputPorts.length; ++j)
                        object.inputPorts[j] = $root.flowcraft.v1.Port.toObject(message.inputPorts[j], options);
                }
                if (message.outputPorts && message.outputPorts.length) {
                    object.outputPorts = [];
                    for (let j = 0; j < message.outputPorts.length; ++j)
                        object.outputPorts[j] = $root.flowcraft.v1.Port.toObject(message.outputPorts[j], options);
                }
                let keys2;
                if (message.metadata && (keys2 = Object.keys(message.metadata)).length) {
                    object.metadata = {};
                    for (let j = 0; j < keys2.length; ++j)
                        object.metadata[keys2[j]] = message.metadata[keys2[j]];
                }
                return object;
            };

            /**
             * Converts this NodeData to JSON.
             * @function toJSON
             * @memberof flowcraft.v1.NodeData
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            NodeData.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };

            /**
             * Gets the default type url for NodeData
             * @function getTypeUrl
             * @memberof flowcraft.v1.NodeData
             * @static
             * @param {string} [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns {string} The default type url
             */
            NodeData.getTypeUrl = function getTypeUrl(typeUrlPrefix) {
                if (typeUrlPrefix === undefined) {
                    typeUrlPrefix = "type.googleapis.com";
                }
                return typeUrlPrefix + "/flowcraft.v1.NodeData";
            };

            return NodeData;
        })();

        /**
         * RenderMode enum.
         * @name flowcraft.v1.RenderMode
         * @enum {number}
         * @property {number} MODE_UNSPECIFIED=0 MODE_UNSPECIFIED value
         * @property {number} MODE_MEDIA=1 MODE_MEDIA value
         * @property {number} MODE_WIDGETS=2 MODE_WIDGETS value
         * @property {number} MODE_MARKDOWN=3 MODE_MARKDOWN value
         */
        v1.RenderMode = (function() {
            const valuesById = {}, values = Object.create(valuesById);
            values[valuesById[0] = "MODE_UNSPECIFIED"] = 0;
            values[valuesById[1] = "MODE_MEDIA"] = 1;
            values[valuesById[2] = "MODE_WIDGETS"] = 2;
            values[valuesById[3] = "MODE_MARKDOWN"] = 3;
            return values;
        })();

        v1.MediaContent = (function() {

            /**
             * Properties of a MediaContent.
             * @memberof flowcraft.v1
             * @interface IMediaContent
             * @property {flowcraft.v1.MediaType|null} [type] MediaContent type
             * @property {string|null} [url] MediaContent url
             * @property {string|null} [content] MediaContent content
             * @property {number|null} [aspectRatio] MediaContent aspectRatio
             * @property {Array.<string>|null} [galleryUrls] MediaContent galleryUrls
             */

            /**
             * Constructs a new MediaContent.
             * @memberof flowcraft.v1
             * @classdesc Represents a MediaContent.
             * @implements IMediaContent
             * @constructor
             * @param {flowcraft.v1.IMediaContent=} [properties] Properties to set
             */
            function MediaContent(properties) {
                this.galleryUrls = [];
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * MediaContent type.
             * @member {flowcraft.v1.MediaType} type
             * @memberof flowcraft.v1.MediaContent
             * @instance
             */
            MediaContent.prototype.type = 0;

            /**
             * MediaContent url.
             * @member {string} url
             * @memberof flowcraft.v1.MediaContent
             * @instance
             */
            MediaContent.prototype.url = "";

            /**
             * MediaContent content.
             * @member {string} content
             * @memberof flowcraft.v1.MediaContent
             * @instance
             */
            MediaContent.prototype.content = "";

            /**
             * MediaContent aspectRatio.
             * @member {number} aspectRatio
             * @memberof flowcraft.v1.MediaContent
             * @instance
             */
            MediaContent.prototype.aspectRatio = 0;

            /**
             * MediaContent galleryUrls.
             * @member {Array.<string>} galleryUrls
             * @memberof flowcraft.v1.MediaContent
             * @instance
             */
            MediaContent.prototype.galleryUrls = $util.emptyArray;

            /**
             * Creates a new MediaContent instance using the specified properties.
             * @function create
             * @memberof flowcraft.v1.MediaContent
             * @static
             * @param {flowcraft.v1.IMediaContent=} [properties] Properties to set
             * @returns {flowcraft.v1.MediaContent} MediaContent instance
             */
            MediaContent.create = function create(properties) {
                return new MediaContent(properties);
            };

            /**
             * Encodes the specified MediaContent message. Does not implicitly {@link flowcraft.v1.MediaContent.verify|verify} messages.
             * @function encode
             * @memberof flowcraft.v1.MediaContent
             * @static
             * @param {flowcraft.v1.IMediaContent} message MediaContent message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            MediaContent.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.type != null && Object.hasOwnProperty.call(message, "type"))
                    writer.uint32(/* id 1, wireType 0 =*/8).int32(message.type);
                if (message.url != null && Object.hasOwnProperty.call(message, "url"))
                    writer.uint32(/* id 2, wireType 2 =*/18).string(message.url);
                if (message.content != null && Object.hasOwnProperty.call(message, "content"))
                    writer.uint32(/* id 3, wireType 2 =*/26).string(message.content);
                if (message.aspectRatio != null && Object.hasOwnProperty.call(message, "aspectRatio"))
                    writer.uint32(/* id 4, wireType 1 =*/33).double(message.aspectRatio);
                if (message.galleryUrls != null && message.galleryUrls.length)
                    for (let i = 0; i < message.galleryUrls.length; ++i)
                        writer.uint32(/* id 5, wireType 2 =*/42).string(message.galleryUrls[i]);
                return writer;
            };

            /**
             * Encodes the specified MediaContent message, length delimited. Does not implicitly {@link flowcraft.v1.MediaContent.verify|verify} messages.
             * @function encodeDelimited
             * @memberof flowcraft.v1.MediaContent
             * @static
             * @param {flowcraft.v1.IMediaContent} message MediaContent message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            MediaContent.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes a MediaContent message from the specified reader or buffer.
             * @function decode
             * @memberof flowcraft.v1.MediaContent
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {flowcraft.v1.MediaContent} MediaContent
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            MediaContent.decode = function decode(reader, length, error) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.flowcraft.v1.MediaContent();
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    if (tag === error)
                        break;
                    switch (tag >>> 3) {
                    case 1: {
                            message.type = reader.int32();
                            break;
                        }
                    case 2: {
                            message.url = reader.string();
                            break;
                        }
                    case 3: {
                            message.content = reader.string();
                            break;
                        }
                    case 4: {
                            message.aspectRatio = reader.double();
                            break;
                        }
                    case 5: {
                            if (!(message.galleryUrls && message.galleryUrls.length))
                                message.galleryUrls = [];
                            message.galleryUrls.push(reader.string());
                            break;
                        }
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };

            /**
             * Decodes a MediaContent message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof flowcraft.v1.MediaContent
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {flowcraft.v1.MediaContent} MediaContent
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            MediaContent.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies a MediaContent message.
             * @function verify
             * @memberof flowcraft.v1.MediaContent
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            MediaContent.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.type != null && message.hasOwnProperty("type"))
                    switch (message.type) {
                    default:
                        return "type: enum value expected";
                    case 0:
                    case 1:
                    case 2:
                    case 3:
                    case 4:
                        break;
                    }
                if (message.url != null && message.hasOwnProperty("url"))
                    if (!$util.isString(message.url))
                        return "url: string expected";
                if (message.content != null && message.hasOwnProperty("content"))
                    if (!$util.isString(message.content))
                        return "content: string expected";
                if (message.aspectRatio != null && message.hasOwnProperty("aspectRatio"))
                    if (typeof message.aspectRatio !== "number")
                        return "aspectRatio: number expected";
                if (message.galleryUrls != null && message.hasOwnProperty("galleryUrls")) {
                    if (!Array.isArray(message.galleryUrls))
                        return "galleryUrls: array expected";
                    for (let i = 0; i < message.galleryUrls.length; ++i)
                        if (!$util.isString(message.galleryUrls[i]))
                            return "galleryUrls: string[] expected";
                }
                return null;
            };

            /**
             * Creates a MediaContent message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof flowcraft.v1.MediaContent
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {flowcraft.v1.MediaContent} MediaContent
             */
            MediaContent.fromObject = function fromObject(object) {
                if (object instanceof $root.flowcraft.v1.MediaContent)
                    return object;
                let message = new $root.flowcraft.v1.MediaContent();
                switch (object.type) {
                default:
                    if (typeof object.type === "number") {
                        message.type = object.type;
                        break;
                    }
                    break;
                case "MEDIA_UNSPECIFIED":
                case 0:
                    message.type = 0;
                    break;
                case "MEDIA_IMAGE":
                case 1:
                    message.type = 1;
                    break;
                case "MEDIA_VIDEO":
                case 2:
                    message.type = 2;
                    break;
                case "MEDIA_AUDIO":
                case 3:
                    message.type = 3;
                    break;
                case "MEDIA_MARKDOWN":
                case 4:
                    message.type = 4;
                    break;
                }
                if (object.url != null)
                    message.url = String(object.url);
                if (object.content != null)
                    message.content = String(object.content);
                if (object.aspectRatio != null)
                    message.aspectRatio = Number(object.aspectRatio);
                if (object.galleryUrls) {
                    if (!Array.isArray(object.galleryUrls))
                        throw TypeError(".flowcraft.v1.MediaContent.galleryUrls: array expected");
                    message.galleryUrls = [];
                    for (let i = 0; i < object.galleryUrls.length; ++i)
                        message.galleryUrls[i] = String(object.galleryUrls[i]);
                }
                return message;
            };

            /**
             * Creates a plain object from a MediaContent message. Also converts values to other types if specified.
             * @function toObject
             * @memberof flowcraft.v1.MediaContent
             * @static
             * @param {flowcraft.v1.MediaContent} message MediaContent
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            MediaContent.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                let object = {};
                if (options.arrays || options.defaults)
                    object.galleryUrls = [];
                if (options.defaults) {
                    object.type = options.enums === String ? "MEDIA_UNSPECIFIED" : 0;
                    object.url = "";
                    object.content = "";
                    object.aspectRatio = 0;
                }
                if (message.type != null && message.hasOwnProperty("type"))
                    object.type = options.enums === String ? $root.flowcraft.v1.MediaType[message.type] === undefined ? message.type : $root.flowcraft.v1.MediaType[message.type] : message.type;
                if (message.url != null && message.hasOwnProperty("url"))
                    object.url = message.url;
                if (message.content != null && message.hasOwnProperty("content"))
                    object.content = message.content;
                if (message.aspectRatio != null && message.hasOwnProperty("aspectRatio"))
                    object.aspectRatio = options.json && !isFinite(message.aspectRatio) ? String(message.aspectRatio) : message.aspectRatio;
                if (message.galleryUrls && message.galleryUrls.length) {
                    object.galleryUrls = [];
                    for (let j = 0; j < message.galleryUrls.length; ++j)
                        object.galleryUrls[j] = message.galleryUrls[j];
                }
                return object;
            };

            /**
             * Converts this MediaContent to JSON.
             * @function toJSON
             * @memberof flowcraft.v1.MediaContent
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            MediaContent.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };

            /**
             * Gets the default type url for MediaContent
             * @function getTypeUrl
             * @memberof flowcraft.v1.MediaContent
             * @static
             * @param {string} [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns {string} The default type url
             */
            MediaContent.getTypeUrl = function getTypeUrl(typeUrlPrefix) {
                if (typeUrlPrefix === undefined) {
                    typeUrlPrefix = "type.googleapis.com";
                }
                return typeUrlPrefix + "/flowcraft.v1.MediaContent";
            };

            return MediaContent;
        })();

        /**
         * MediaType enum.
         * @name flowcraft.v1.MediaType
         * @enum {number}
         * @property {number} MEDIA_UNSPECIFIED=0 MEDIA_UNSPECIFIED value
         * @property {number} MEDIA_IMAGE=1 MEDIA_IMAGE value
         * @property {number} MEDIA_VIDEO=2 MEDIA_VIDEO value
         * @property {number} MEDIA_AUDIO=3 MEDIA_AUDIO value
         * @property {number} MEDIA_MARKDOWN=4 MEDIA_MARKDOWN value
         */
        v1.MediaType = (function() {
            const valuesById = {}, values = Object.create(valuesById);
            values[valuesById[0] = "MEDIA_UNSPECIFIED"] = 0;
            values[valuesById[1] = "MEDIA_IMAGE"] = 1;
            values[valuesById[2] = "MEDIA_VIDEO"] = 2;
            values[valuesById[3] = "MEDIA_AUDIO"] = 3;
            values[valuesById[4] = "MEDIA_MARKDOWN"] = 4;
            return values;
        })();

        /**
         * WidgetType enum.
         * @name flowcraft.v1.WidgetType
         * @enum {number}
         * @property {number} WIDGET_UNSPECIFIED=0 WIDGET_UNSPECIFIED value
         * @property {number} WIDGET_TEXT=1 WIDGET_TEXT value
         * @property {number} WIDGET_SELECT=2 WIDGET_SELECT value
         * @property {number} WIDGET_CHECKBOX=3 WIDGET_CHECKBOX value
         * @property {number} WIDGET_SLIDER=4 WIDGET_SLIDER value
         * @property {number} WIDGET_BUTTON=5 WIDGET_BUTTON value
         */
        v1.WidgetType = (function() {
            const valuesById = {}, values = Object.create(valuesById);
            values[valuesById[0] = "WIDGET_UNSPECIFIED"] = 0;
            values[valuesById[1] = "WIDGET_TEXT"] = 1;
            values[valuesById[2] = "WIDGET_SELECT"] = 2;
            values[valuesById[3] = "WIDGET_CHECKBOX"] = 3;
            values[valuesById[4] = "WIDGET_SLIDER"] = 4;
            values[valuesById[5] = "WIDGET_BUTTON"] = 5;
            return values;
        })();

        v1.WidgetOption = (function() {

            /**
             * Properties of a WidgetOption.
             * @memberof flowcraft.v1
             * @interface IWidgetOption
             * @property {string|null} [label] WidgetOption label
             * @property {string|null} [value] WidgetOption value
             * @property {string|null} [description] WidgetOption description
             */

            /**
             * Constructs a new WidgetOption.
             * @memberof flowcraft.v1
             * @classdesc Represents a WidgetOption.
             * @implements IWidgetOption
             * @constructor
             * @param {flowcraft.v1.IWidgetOption=} [properties] Properties to set
             */
            function WidgetOption(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * WidgetOption label.
             * @member {string} label
             * @memberof flowcraft.v1.WidgetOption
             * @instance
             */
            WidgetOption.prototype.label = "";

            /**
             * WidgetOption value.
             * @member {string} value
             * @memberof flowcraft.v1.WidgetOption
             * @instance
             */
            WidgetOption.prototype.value = "";

            /**
             * WidgetOption description.
             * @member {string} description
             * @memberof flowcraft.v1.WidgetOption
             * @instance
             */
            WidgetOption.prototype.description = "";

            /**
             * Creates a new WidgetOption instance using the specified properties.
             * @function create
             * @memberof flowcraft.v1.WidgetOption
             * @static
             * @param {flowcraft.v1.IWidgetOption=} [properties] Properties to set
             * @returns {flowcraft.v1.WidgetOption} WidgetOption instance
             */
            WidgetOption.create = function create(properties) {
                return new WidgetOption(properties);
            };

            /**
             * Encodes the specified WidgetOption message. Does not implicitly {@link flowcraft.v1.WidgetOption.verify|verify} messages.
             * @function encode
             * @memberof flowcraft.v1.WidgetOption
             * @static
             * @param {flowcraft.v1.IWidgetOption} message WidgetOption message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            WidgetOption.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.label != null && Object.hasOwnProperty.call(message, "label"))
                    writer.uint32(/* id 1, wireType 2 =*/10).string(message.label);
                if (message.value != null && Object.hasOwnProperty.call(message, "value"))
                    writer.uint32(/* id 2, wireType 2 =*/18).string(message.value);
                if (message.description != null && Object.hasOwnProperty.call(message, "description"))
                    writer.uint32(/* id 3, wireType 2 =*/26).string(message.description);
                return writer;
            };

            /**
             * Encodes the specified WidgetOption message, length delimited. Does not implicitly {@link flowcraft.v1.WidgetOption.verify|verify} messages.
             * @function encodeDelimited
             * @memberof flowcraft.v1.WidgetOption
             * @static
             * @param {flowcraft.v1.IWidgetOption} message WidgetOption message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            WidgetOption.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes a WidgetOption message from the specified reader or buffer.
             * @function decode
             * @memberof flowcraft.v1.WidgetOption
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {flowcraft.v1.WidgetOption} WidgetOption
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            WidgetOption.decode = function decode(reader, length, error) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.flowcraft.v1.WidgetOption();
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    if (tag === error)
                        break;
                    switch (tag >>> 3) {
                    case 1: {
                            message.label = reader.string();
                            break;
                        }
                    case 2: {
                            message.value = reader.string();
                            break;
                        }
                    case 3: {
                            message.description = reader.string();
                            break;
                        }
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };

            /**
             * Decodes a WidgetOption message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof flowcraft.v1.WidgetOption
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {flowcraft.v1.WidgetOption} WidgetOption
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            WidgetOption.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies a WidgetOption message.
             * @function verify
             * @memberof flowcraft.v1.WidgetOption
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            WidgetOption.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.label != null && message.hasOwnProperty("label"))
                    if (!$util.isString(message.label))
                        return "label: string expected";
                if (message.value != null && message.hasOwnProperty("value"))
                    if (!$util.isString(message.value))
                        return "value: string expected";
                if (message.description != null && message.hasOwnProperty("description"))
                    if (!$util.isString(message.description))
                        return "description: string expected";
                return null;
            };

            /**
             * Creates a WidgetOption message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof flowcraft.v1.WidgetOption
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {flowcraft.v1.WidgetOption} WidgetOption
             */
            WidgetOption.fromObject = function fromObject(object) {
                if (object instanceof $root.flowcraft.v1.WidgetOption)
                    return object;
                let message = new $root.flowcraft.v1.WidgetOption();
                if (object.label != null)
                    message.label = String(object.label);
                if (object.value != null)
                    message.value = String(object.value);
                if (object.description != null)
                    message.description = String(object.description);
                return message;
            };

            /**
             * Creates a plain object from a WidgetOption message. Also converts values to other types if specified.
             * @function toObject
             * @memberof flowcraft.v1.WidgetOption
             * @static
             * @param {flowcraft.v1.WidgetOption} message WidgetOption
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            WidgetOption.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                let object = {};
                if (options.defaults) {
                    object.label = "";
                    object.value = "";
                    object.description = "";
                }
                if (message.label != null && message.hasOwnProperty("label"))
                    object.label = message.label;
                if (message.value != null && message.hasOwnProperty("value"))
                    object.value = message.value;
                if (message.description != null && message.hasOwnProperty("description"))
                    object.description = message.description;
                return object;
            };

            /**
             * Converts this WidgetOption to JSON.
             * @function toJSON
             * @memberof flowcraft.v1.WidgetOption
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            WidgetOption.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };

            /**
             * Gets the default type url for WidgetOption
             * @function getTypeUrl
             * @memberof flowcraft.v1.WidgetOption
             * @static
             * @param {string} [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns {string} The default type url
             */
            WidgetOption.getTypeUrl = function getTypeUrl(typeUrlPrefix) {
                if (typeUrlPrefix === undefined) {
                    typeUrlPrefix = "type.googleapis.com";
                }
                return typeUrlPrefix + "/flowcraft.v1.WidgetOption";
            };

            return WidgetOption;
        })();

        v1.WidgetConfig = (function() {

            /**
             * Properties of a WidgetConfig.
             * @memberof flowcraft.v1
             * @interface IWidgetConfig
             * @property {string|null} [placeholder] WidgetConfig placeholder
             * @property {number|null} [min] WidgetConfig min
             * @property {number|null} [max] WidgetConfig max
             * @property {number|null} [step] WidgetConfig step
             * @property {boolean|null} [dynamicOptions] WidgetConfig dynamicOptions
             * @property {string|null} [actionTarget] WidgetConfig actionTarget
             */

            /**
             * Constructs a new WidgetConfig.
             * @memberof flowcraft.v1
             * @classdesc Represents a WidgetConfig.
             * @implements IWidgetConfig
             * @constructor
             * @param {flowcraft.v1.IWidgetConfig=} [properties] Properties to set
             */
            function WidgetConfig(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * WidgetConfig placeholder.
             * @member {string} placeholder
             * @memberof flowcraft.v1.WidgetConfig
             * @instance
             */
            WidgetConfig.prototype.placeholder = "";

            /**
             * WidgetConfig min.
             * @member {number} min
             * @memberof flowcraft.v1.WidgetConfig
             * @instance
             */
            WidgetConfig.prototype.min = 0;

            /**
             * WidgetConfig max.
             * @member {number} max
             * @memberof flowcraft.v1.WidgetConfig
             * @instance
             */
            WidgetConfig.prototype.max = 0;

            /**
             * WidgetConfig step.
             * @member {number} step
             * @memberof flowcraft.v1.WidgetConfig
             * @instance
             */
            WidgetConfig.prototype.step = 0;

            /**
             * WidgetConfig dynamicOptions.
             * @member {boolean} dynamicOptions
             * @memberof flowcraft.v1.WidgetConfig
             * @instance
             */
            WidgetConfig.prototype.dynamicOptions = false;

            /**
             * WidgetConfig actionTarget.
             * @member {string} actionTarget
             * @memberof flowcraft.v1.WidgetConfig
             * @instance
             */
            WidgetConfig.prototype.actionTarget = "";

            /**
             * Creates a new WidgetConfig instance using the specified properties.
             * @function create
             * @memberof flowcraft.v1.WidgetConfig
             * @static
             * @param {flowcraft.v1.IWidgetConfig=} [properties] Properties to set
             * @returns {flowcraft.v1.WidgetConfig} WidgetConfig instance
             */
            WidgetConfig.create = function create(properties) {
                return new WidgetConfig(properties);
            };

            /**
             * Encodes the specified WidgetConfig message. Does not implicitly {@link flowcraft.v1.WidgetConfig.verify|verify} messages.
             * @function encode
             * @memberof flowcraft.v1.WidgetConfig
             * @static
             * @param {flowcraft.v1.IWidgetConfig} message WidgetConfig message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            WidgetConfig.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.placeholder != null && Object.hasOwnProperty.call(message, "placeholder"))
                    writer.uint32(/* id 1, wireType 2 =*/10).string(message.placeholder);
                if (message.min != null && Object.hasOwnProperty.call(message, "min"))
                    writer.uint32(/* id 2, wireType 1 =*/17).double(message.min);
                if (message.max != null && Object.hasOwnProperty.call(message, "max"))
                    writer.uint32(/* id 3, wireType 1 =*/25).double(message.max);
                if (message.step != null && Object.hasOwnProperty.call(message, "step"))
                    writer.uint32(/* id 4, wireType 1 =*/33).double(message.step);
                if (message.dynamicOptions != null && Object.hasOwnProperty.call(message, "dynamicOptions"))
                    writer.uint32(/* id 5, wireType 0 =*/40).bool(message.dynamicOptions);
                if (message.actionTarget != null && Object.hasOwnProperty.call(message, "actionTarget"))
                    writer.uint32(/* id 6, wireType 2 =*/50).string(message.actionTarget);
                return writer;
            };

            /**
             * Encodes the specified WidgetConfig message, length delimited. Does not implicitly {@link flowcraft.v1.WidgetConfig.verify|verify} messages.
             * @function encodeDelimited
             * @memberof flowcraft.v1.WidgetConfig
             * @static
             * @param {flowcraft.v1.IWidgetConfig} message WidgetConfig message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            WidgetConfig.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes a WidgetConfig message from the specified reader or buffer.
             * @function decode
             * @memberof flowcraft.v1.WidgetConfig
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {flowcraft.v1.WidgetConfig} WidgetConfig
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            WidgetConfig.decode = function decode(reader, length, error) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.flowcraft.v1.WidgetConfig();
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    if (tag === error)
                        break;
                    switch (tag >>> 3) {
                    case 1: {
                            message.placeholder = reader.string();
                            break;
                        }
                    case 2: {
                            message.min = reader.double();
                            break;
                        }
                    case 3: {
                            message.max = reader.double();
                            break;
                        }
                    case 4: {
                            message.step = reader.double();
                            break;
                        }
                    case 5: {
                            message.dynamicOptions = reader.bool();
                            break;
                        }
                    case 6: {
                            message.actionTarget = reader.string();
                            break;
                        }
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };

            /**
             * Decodes a WidgetConfig message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof flowcraft.v1.WidgetConfig
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {flowcraft.v1.WidgetConfig} WidgetConfig
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            WidgetConfig.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies a WidgetConfig message.
             * @function verify
             * @memberof flowcraft.v1.WidgetConfig
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            WidgetConfig.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.placeholder != null && message.hasOwnProperty("placeholder"))
                    if (!$util.isString(message.placeholder))
                        return "placeholder: string expected";
                if (message.min != null && message.hasOwnProperty("min"))
                    if (typeof message.min !== "number")
                        return "min: number expected";
                if (message.max != null && message.hasOwnProperty("max"))
                    if (typeof message.max !== "number")
                        return "max: number expected";
                if (message.step != null && message.hasOwnProperty("step"))
                    if (typeof message.step !== "number")
                        return "step: number expected";
                if (message.dynamicOptions != null && message.hasOwnProperty("dynamicOptions"))
                    if (typeof message.dynamicOptions !== "boolean")
                        return "dynamicOptions: boolean expected";
                if (message.actionTarget != null && message.hasOwnProperty("actionTarget"))
                    if (!$util.isString(message.actionTarget))
                        return "actionTarget: string expected";
                return null;
            };

            /**
             * Creates a WidgetConfig message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof flowcraft.v1.WidgetConfig
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {flowcraft.v1.WidgetConfig} WidgetConfig
             */
            WidgetConfig.fromObject = function fromObject(object) {
                if (object instanceof $root.flowcraft.v1.WidgetConfig)
                    return object;
                let message = new $root.flowcraft.v1.WidgetConfig();
                if (object.placeholder != null)
                    message.placeholder = String(object.placeholder);
                if (object.min != null)
                    message.min = Number(object.min);
                if (object.max != null)
                    message.max = Number(object.max);
                if (object.step != null)
                    message.step = Number(object.step);
                if (object.dynamicOptions != null)
                    message.dynamicOptions = Boolean(object.dynamicOptions);
                if (object.actionTarget != null)
                    message.actionTarget = String(object.actionTarget);
                return message;
            };

            /**
             * Creates a plain object from a WidgetConfig message. Also converts values to other types if specified.
             * @function toObject
             * @memberof flowcraft.v1.WidgetConfig
             * @static
             * @param {flowcraft.v1.WidgetConfig} message WidgetConfig
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            WidgetConfig.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                let object = {};
                if (options.defaults) {
                    object.placeholder = "";
                    object.min = 0;
                    object.max = 0;
                    object.step = 0;
                    object.dynamicOptions = false;
                    object.actionTarget = "";
                }
                if (message.placeholder != null && message.hasOwnProperty("placeholder"))
                    object.placeholder = message.placeholder;
                if (message.min != null && message.hasOwnProperty("min"))
                    object.min = options.json && !isFinite(message.min) ? String(message.min) : message.min;
                if (message.max != null && message.hasOwnProperty("max"))
                    object.max = options.json && !isFinite(message.max) ? String(message.max) : message.max;
                if (message.step != null && message.hasOwnProperty("step"))
                    object.step = options.json && !isFinite(message.step) ? String(message.step) : message.step;
                if (message.dynamicOptions != null && message.hasOwnProperty("dynamicOptions"))
                    object.dynamicOptions = message.dynamicOptions;
                if (message.actionTarget != null && message.hasOwnProperty("actionTarget"))
                    object.actionTarget = message.actionTarget;
                return object;
            };

            /**
             * Converts this WidgetConfig to JSON.
             * @function toJSON
             * @memberof flowcraft.v1.WidgetConfig
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            WidgetConfig.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };

            /**
             * Gets the default type url for WidgetConfig
             * @function getTypeUrl
             * @memberof flowcraft.v1.WidgetConfig
             * @static
             * @param {string} [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns {string} The default type url
             */
            WidgetConfig.getTypeUrl = function getTypeUrl(typeUrlPrefix) {
                if (typeUrlPrefix === undefined) {
                    typeUrlPrefix = "type.googleapis.com";
                }
                return typeUrlPrefix + "/flowcraft.v1.WidgetConfig";
            };

            return WidgetConfig;
        })();

        v1.Widget = (function() {

            /**
             * Properties of a Widget.
             * @memberof flowcraft.v1
             * @interface IWidget
             * @property {string|null} [id] Widget id
             * @property {flowcraft.v1.WidgetType|null} [type] Widget type
             * @property {string|null} [label] Widget label
             * @property {string|null} [valueJson] Widget valueJson
             * @property {flowcraft.v1.IWidgetConfig|null} [config] Widget config
             * @property {Array.<flowcraft.v1.IWidgetOption>|null} [options] Widget options
             * @property {boolean|null} [isReadonly] Widget isReadonly
             * @property {boolean|null} [isLoading] Widget isLoading
             * @property {string|null} [inputPortId] Widget inputPortId
             */

            /**
             * Constructs a new Widget.
             * @memberof flowcraft.v1
             * @classdesc Represents a Widget.
             * @implements IWidget
             * @constructor
             * @param {flowcraft.v1.IWidget=} [properties] Properties to set
             */
            function Widget(properties) {
                this.options = [];
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null)
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * Widget id.
             * @member {string} id
             * @memberof flowcraft.v1.Widget
             * @instance
             */
            Widget.prototype.id = "";

            /**
             * Widget type.
             * @member {flowcraft.v1.WidgetType} type
             * @memberof flowcraft.v1.Widget
             * @instance
             */
            Widget.prototype.type = 0;

            /**
             * Widget label.
             * @member {string} label
             * @memberof flowcraft.v1.Widget
             * @instance
             */
            Widget.prototype.label = "";

            /**
             * Widget valueJson.
             * @member {string} valueJson
             * @memberof flowcraft.v1.Widget
             * @instance
             */
            Widget.prototype.valueJson = "";

            /**
             * Widget config.
             * @member {flowcraft.v1.IWidgetConfig|null|undefined} config
             * @memberof flowcraft.v1.Widget
             * @instance
             */
            Widget.prototype.config = null;

            /**
             * Widget options.
             * @member {Array.<flowcraft.v1.IWidgetOption>} options
             * @memberof flowcraft.v1.Widget
             * @instance
             */
            Widget.prototype.options = $util.emptyArray;

            /**
             * Widget isReadonly.
             * @member {boolean} isReadonly
             * @memberof flowcraft.v1.Widget
             * @instance
             */
            Widget.prototype.isReadonly = false;

            /**
             * Widget isLoading.
             * @member {boolean} isLoading
             * @memberof flowcraft.v1.Widget
             * @instance
             */
            Widget.prototype.isLoading = false;

            /**
             * Widget inputPortId.
             * @member {string} inputPortId
             * @memberof flowcraft.v1.Widget
             * @instance
             */
            Widget.prototype.inputPortId = "";

            /**
             * Creates a new Widget instance using the specified properties.
             * @function create
             * @memberof flowcraft.v1.Widget
             * @static
             * @param {flowcraft.v1.IWidget=} [properties] Properties to set
             * @returns {flowcraft.v1.Widget} Widget instance
             */
            Widget.create = function create(properties) {
                return new Widget(properties);
            };

            /**
             * Encodes the specified Widget message. Does not implicitly {@link flowcraft.v1.Widget.verify|verify} messages.
             * @function encode
             * @memberof flowcraft.v1.Widget
             * @static
             * @param {flowcraft.v1.IWidget} message Widget message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            Widget.encode = function encode(message, writer) {
                if (!writer)
                    writer = $Writer.create();
                if (message.id != null && Object.hasOwnProperty.call(message, "id"))
                    writer.uint32(/* id 1, wireType 2 =*/10).string(message.id);
                if (message.type != null && Object.hasOwnProperty.call(message, "type"))
                    writer.uint32(/* id 2, wireType 0 =*/16).int32(message.type);
                if (message.label != null && Object.hasOwnProperty.call(message, "label"))
                    writer.uint32(/* id 3, wireType 2 =*/26).string(message.label);
                if (message.valueJson != null && Object.hasOwnProperty.call(message, "valueJson"))
                    writer.uint32(/* id 4, wireType 2 =*/34).string(message.valueJson);
                if (message.config != null && Object.hasOwnProperty.call(message, "config"))
                    $root.flowcraft.v1.WidgetConfig.encode(message.config, writer.uint32(/* id 5, wireType 2 =*/42).fork()).ldelim();
                if (message.options != null && message.options.length)
                    for (let i = 0; i < message.options.length; ++i)
                        $root.flowcraft.v1.WidgetOption.encode(message.options[i], writer.uint32(/* id 6, wireType 2 =*/50).fork()).ldelim();
                if (message.isReadonly != null && Object.hasOwnProperty.call(message, "isReadonly"))
                    writer.uint32(/* id 7, wireType 0 =*/56).bool(message.isReadonly);
                if (message.isLoading != null && Object.hasOwnProperty.call(message, "isLoading"))
                    writer.uint32(/* id 8, wireType 0 =*/64).bool(message.isLoading);
                if (message.inputPortId != null && Object.hasOwnProperty.call(message, "inputPortId"))
                    writer.uint32(/* id 9, wireType 2 =*/74).string(message.inputPortId);
                return writer;
            };

            /**
             * Encodes the specified Widget message, length delimited. Does not implicitly {@link flowcraft.v1.Widget.verify|verify} messages.
             * @function encodeDelimited
             * @memberof flowcraft.v1.Widget
             * @static
             * @param {flowcraft.v1.IWidget} message Widget message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            Widget.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer).ldelim();
            };

            /**
             * Decodes a Widget message from the specified reader or buffer.
             * @function decode
             * @memberof flowcraft.v1.Widget
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {flowcraft.v1.Widget} Widget
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            Widget.decode = function decode(reader, length, error) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.flowcraft.v1.Widget();
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    if (tag === error)
                        break;
                    switch (tag >>> 3) {
                    case 1: {
                            message.id = reader.string();
                            break;
                        }
                    case 2: {
                            message.type = reader.int32();
                            break;
                        }
                    case 3: {
                            message.label = reader.string();
                            break;
                        }
                    case 4: {
                            message.valueJson = reader.string();
                            break;
                        }
                    case 5: {
                            message.config = $root.flowcraft.v1.WidgetConfig.decode(reader, reader.uint32());
                            break;
                        }
                    case 6: {
                            if (!(message.options && message.options.length))
                                message.options = [];
                            message.options.push($root.flowcraft.v1.WidgetOption.decode(reader, reader.uint32()));
                            break;
                        }
                    case 7: {
                            message.isReadonly = reader.bool();
                            break;
                        }
                    case 8: {
                            message.isLoading = reader.bool();
                            break;
                        }
                    case 9: {
                            message.inputPortId = reader.string();
                            break;
                        }
                    default:
                        reader.skipType(tag & 7);
                        break;
                    }
                }
                return message;
            };

            /**
             * Decodes a Widget message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof flowcraft.v1.Widget
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {flowcraft.v1.Widget} Widget
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            Widget.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies a Widget message.
             * @function verify
             * @memberof flowcraft.v1.Widget
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            Widget.verify = function verify(message) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (message.id != null && message.hasOwnProperty("id"))
                    if (!$util.isString(message.id))
                        return "id: string expected";
                if (message.type != null && message.hasOwnProperty("type"))
                    switch (message.type) {
                    default:
                        return "type: enum value expected";
                    case 0:
                    case 1:
                    case 2:
                    case 3:
                    case 4:
                    case 5:
                        break;
                    }
                if (message.label != null && message.hasOwnProperty("label"))
                    if (!$util.isString(message.label))
                        return "label: string expected";
                if (message.valueJson != null && message.hasOwnProperty("valueJson"))
                    if (!$util.isString(message.valueJson))
                        return "valueJson: string expected";
                if (message.config != null && message.hasOwnProperty("config")) {
                    let error = $root.flowcraft.v1.WidgetConfig.verify(message.config);
                    if (error)
                        return "config." + error;
                }
                if (message.options != null && message.hasOwnProperty("options")) {
                    if (!Array.isArray(message.options))
                        return "options: array expected";
                    for (let i = 0; i < message.options.length; ++i) {
                        let error = $root.flowcraft.v1.WidgetOption.verify(message.options[i]);
                        if (error)
                            return "options." + error;
                    }
                }
                if (message.isReadonly != null && message.hasOwnProperty("isReadonly"))
                    if (typeof message.isReadonly !== "boolean")
                        return "isReadonly: boolean expected";
                if (message.isLoading != null && message.hasOwnProperty("isLoading"))
                    if (typeof message.isLoading !== "boolean")
                        return "isLoading: boolean expected";
                if (message.inputPortId != null && message.hasOwnProperty("inputPortId"))
                    if (!$util.isString(message.inputPortId))
                        return "inputPortId: string expected";
                return null;
            };

            /**
             * Creates a Widget message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof flowcraft.v1.Widget
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {flowcraft.v1.Widget} Widget
             */
            Widget.fromObject = function fromObject(object) {
                if (object instanceof $root.flowcraft.v1.Widget)
                    return object;
                let message = new $root.flowcraft.v1.Widget();
                if (object.id != null)
                    message.id = String(object.id);
                switch (object.type) {
                default:
                    if (typeof object.type === "number") {
                        message.type = object.type;
                        break;
                    }
                    break;
                case "WIDGET_UNSPECIFIED":
                case 0:
                    message.type = 0;
                    break;
                case "WIDGET_TEXT":
                case 1:
                    message.type = 1;
                    break;
                case "WIDGET_SELECT":
                case 2:
                    message.type = 2;
                    break;
                case "WIDGET_CHECKBOX":
                case 3:
                    message.type = 3;
                    break;
                case "WIDGET_SLIDER":
                case 4:
                    message.type = 4;
                    break;
                case "WIDGET_BUTTON":
                case 5:
                    message.type = 5;
                    break;
                }
                if (object.label != null)
                    message.label = String(object.label);
                if (object.valueJson != null)
                    message.valueJson = String(object.valueJson);
                if (object.config != null) {
                    if (typeof object.config !== "object")
                        throw TypeError(".flowcraft.v1.Widget.config: object expected");
                    message.config = $root.flowcraft.v1.WidgetConfig.fromObject(object.config);
                }
                if (object.options) {
                    if (!Array.isArray(object.options))
                        throw TypeError(".flowcraft.v1.Widget.options: array expected");
                    message.options = [];
                    for (let i = 0; i < object.options.length; ++i) {
                        if (typeof object.options[i] !== "object")
                            throw TypeError(".flowcraft.v1.Widget.options: object expected");
                        message.options[i] = $root.flowcraft.v1.WidgetOption.fromObject(object.options[i]);
                    }
                }
                if (object.isReadonly != null)
                    message.isReadonly = Boolean(object.isReadonly);
                if (object.isLoading != null)
                    message.isLoading = Boolean(object.isLoading);
                if (object.inputPortId != null)
                    message.inputPortId = String(object.inputPortId);
                return message;
            };

            /**
             * Creates a plain object from a Widget message. Also converts values to other types if specified.
             * @function toObject
             * @memberof flowcraft.v1.Widget
             * @static
             * @param {flowcraft.v1.Widget} message Widget
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            Widget.toObject = function toObject(message, options) {
                if (!options)
                    options = {};
                let object = {};
                if (options.arrays || options.defaults)
                    object.options = [];
                if (options.defaults) {
                    object.id = "";
                    object.type = options.enums === String ? "WIDGET_UNSPECIFIED" : 0;
                    object.label = "";
                    object.valueJson = "";
                    object.config = null;
                    object.isReadonly = false;
                    object.isLoading = false;
                    object.inputPortId = "";
                }
                if (message.id != null && message.hasOwnProperty("id"))
                    object.id = message.id;
                if (message.type != null && message.hasOwnProperty("type"))
                    object.type = options.enums === String ? $root.flowcraft.v1.WidgetType[message.type] === undefined ? message.type : $root.flowcraft.v1.WidgetType[message.type] : message.type;
                if (message.label != null && message.hasOwnProperty("label"))
                    object.label = message.label;
                if (message.valueJson != null && message.hasOwnProperty("valueJson"))
                    object.valueJson = message.valueJson;
                if (message.config != null && message.hasOwnProperty("config"))
                    object.config = $root.flowcraft.v1.WidgetConfig.toObject(message.config, options);
                if (message.options && message.options.length) {
                    object.options = [];
                    for (let j = 0; j < message.options.length; ++j)
                        object.options[j] = $root.flowcraft.v1.WidgetOption.toObject(message.options[j], options);
                }
                if (message.isReadonly != null && message.hasOwnProperty("isReadonly"))
                    object.isReadonly = message.isReadonly;
                if (message.isLoading != null && message.hasOwnProperty("isLoading"))
                    object.isLoading = message.isLoading;
                if (message.inputPortId != null && message.hasOwnProperty("inputPortId"))
                    object.inputPortId = message.inputPortId;
                return object;
            };

            /**
             * Converts this Widget to JSON.
             * @function toJSON
             * @memberof flowcraft.v1.Widget
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            Widget.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };

            /**
             * Gets the default type url for Widget
             * @function getTypeUrl
             * @memberof flowcraft.v1.Widget
             * @static
             * @param {string} [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns {string} The default type url
             */
            Widget.getTypeUrl = function getTypeUrl(typeUrlPrefix) {
                if (typeUrlPrefix === undefined) {
                    typeUrlPrefix = "type.googleapis.com";
                }
                return typeUrlPrefix + "/flowcraft.v1.Widget";
            };

            return Widget;
        })();

        return v1;
    })();

    return flowcraft;
})();

export { $root as default };
