# Design Specification

This document outlines the design principles and guidelines for developing models, systems, and other architectural components within the ECS Digital Asset Management (DAM) system.

## Table of Contents

- [1. Core Architecture](#1-core-architecture)
- [2. Models (Components)](#2-models-components)
  - [2.1. Definition and Purpose](#21-definition-and-purpose)
  - [2.2. Naming Conventions](#22-naming-conventions)
  - [2.3. Inheritance](#23-inheritance)
  - [2.4. Field Definitions](#24-field-definitions)
  - [2.5. Semantic Embedding Components (Text)](#25-semantic-embedding-components-text)
  - [2.6. Audio Embedding Components](#26-audio-embedding-components)
  - [2.7. Tagging Components and Functions](#27-tagging-components-and-functions)
  - [2.8. Constructor Expectations](#28-constructor-expectations)
- [3. Functions, Systems, Commands, and Resources](#3-functions-systems-commands-and-resources)
  - [3.1. Functions](#31-functions)
  - [3.2. Systems](#32-systems)
  - [3.3. Commands](#33-commands)
  - [3.4. Core Asset Events and Commands](#34-core-asset-events-and-commands)
  - [3.5. Events vs. Commands](#35-events-vs-commands)
  - [3.6. Resources](#36-resources)
- [4. Testing Guidelines](#4-testing-guidelines)
- [5. Further Information](#5-further-information)

## 1. Core Architecture

The system is built upon the Entity-Component-System (ECS) pattern. This pattern promotes flexibility, modularity, and separation of concerns.
- **Entities**: Unique identifiers for assets or concepts.
- **Components**: Data-only classes describing aspects of entities.
- **Systems**: Logic operating on entities based on their components.
- **Plugins**: A modular, Bevy-like plugin system for extending functionality. Plugins are responsible for registering systems and resources with a `World`.

For a comprehensive understanding of the ECS implementation, refer to the [Developer Guide](developer_guide.md).

### 1.1. Architectural Preference: Commands over Events/Stages

A key design principle for the `dam` ecosystem is to **prefer the Command pattern for implementing new functionality**.

-   **Why?**: Commands provide a clear, imperative, and traceable control flow. When you dispatch a command, you have a clear expectation of a specific action being performed. This makes the system easier to understand, debug, and test.
-   **Guideline**: Unless a task's requirements explicitly call for a decoupled, event-driven workflow (e.g., multiple independent systems reacting to a single occurrence) or a lifecycle-based stage, you should implement the logic as a command and its corresponding handler system. Avoid using events or component markers as the primary mechanism for triggering core business logic.

### 1.2. Dependency Rule

A critical design principle is that the core `dam` package **must not** depend on any of its plugin packages (e.g., `dam_media_image`, `dam_sire`). This ensures the core remains lean and decoupled.

-   **Core `dam` Package**: Contains the fundamental ECS framework, base components, and core functions that have no knowledge of specific media types or external model frameworks.
-   **Plugin Packages**: Extend the core functionality. They can depend on `dam`, but `dam` cannot depend on them.
-   **Application Package (`dam_app`)**: The main application that brings everything together. It depends on `dam` and all the necessary plugins.

If a system or function module within the `dam` package needs functionality from a plugin, it must be moved to the appropriate plugin package or to the `dam_app` package if it's application-specific logic.

## 2. Models (Components)

### 2.1. Definition and Purpose
Components are data-only Python classes that store specific attributes or properties of an entity. They should not contain business logic.

### 2.2. Naming Conventions
- **Class Names**: `PascalCase` (e.g., `FilePropertiesComponent`).
- **Convention for Table Names**: Generally `component_[name]` (e.g., `component_file_properties`).
    - For specific embedding components, table names are more descriptive, like `component_embedding_minilm_l6_v2_d384`, indicating the model, version, and key parameters (e.g., dimension).
    - Component class names also reflect this specificity, e.g., `TextEmbeddingAllMiniLML6V2Dim384Component`.

### 2.3. Inheritance
All components inherit from a common abstract base class, `dam.models.core.base_component.Component`. This ensures that all components can be discovered and handled by the core ECS functions. From there, the hierarchy splits based on the component's relationship with its entity:

- **`BaseComponent`**: For components that can have **multiple instances** per entity. This is the most common type of component. It has its own auto-incrementing `id` primary key, in addition to the `entity_id` foreign key.
- **`UniqueComponent`**: For components that must have a **one-to-one relationship** with an entity. This base class enforces uniqueness at the database level by using the `entity_id` as its primary key. This is a more efficient and robust way to handle unique components than the previous `UniqueComponentMixin`.
- **Specialized Base Classes**: More specific, abstract base classes can be created for families of components. For example, `BaseSpecificEmbeddingComponent` inherits from `BaseComponent` and provides common fields for embedding vectors. Similarly, a unique version could be created that inherits from `UniqueComponent`.

### 2.4. Field Definitions
- Component fields should be defined using SQLAlchemy's `Mapped` type hints and `mapped_column` function.
- Example: `original_filename: Mapped[Optional[str]] = mapped_column(String())`
- Specific text embedding tables do *not* store `model_name` or `model_parameters` as columns; this information is inherent in the choice of table. For audio embeddings, `model_name` is stored as a column in the base specific audio embedding component due to the initial design choice, allowing a single table to potentially store embeddings from different audio models if their features are compatible, or to clearly identify the source model if multiple specific tables are used per audio model type.

### 2.5. Semantic Embedding Components (Text)

The system uses a structured approach for managing text embeddings from different models and hyperparameters:

-   **Specific Tables per Model/Configuration**: Each unique combination of an embedding model and its critical hyperparameters (e.g., output dimension) has its own dedicated database table. This avoids storing redundant model information in rows and allows for optimized table structures if needed.
-   **Naming Convention**:
    *   Component Class: `TextEmbedding<ModelName><KeyParam1><Value1>...Component` (e.g., `TextEmbeddingAllMiniLML6V2Dim384Component`).
    *   Table Name: `component_embedding_<model_name_lower>_<key_param1>_<value1>...` (e.g., `component_embedding_minilm_l6_v2_d384`).
-   **Base Class**: All specific embedding components inherit from `BaseSpecificEmbeddingComponent`.
-   **Registry**: A central registry (`EMBEDDING_MODEL_REGISTRY` in `dam.models.semantic.text_embedding_component.py`) maps a user-friendly `model_name` string (and optionally, a dictionary of parameters) to the corresponding SQLAlchemy component class.
    ```python
    EMBEDDING_MODEL_REGISTRY: Dict[str, EmbeddingModelInfo] = {
        "all-MiniLM-L6-v2": {
            "model_class": TextEmbeddingAllMiniLML6V2Dim384Component,
            "default_params": {"dimensions": 384},
        },
        # ... other models
    }
    ```
-   **Functions Layer**: The `semantic_functions.py` uses this registry (`get_embedding_component_class` function) to determine the correct component class (and thus, table) for operations like creating, retrieving, or searching embeddings based on the provided `model_name` and `model_params`.
    The functions also support augmenting input text with tags (manual or model-generated) before generating embeddings. This is controlled by parameters like `include_manual_tags` and `include_model_tags_config` in `update_text_embeddings_for_entity`. The `source_field_name` in the stored embedding component will be suffixed (e.g., `_with_tags`) to indicate this augmentation.

#### Adding a New Embedding Type/Model

To support a new embedding model or a new configuration of an existing model that warrants its own table:

1.  **Define the Component Class**:
    *   Create a new class in `dam/models/semantic/text_embedding_component.py`.
    *   It must inherit from `BaseSpecificEmbeddingComponent`.
    *   Set the `__tablename__` according to the naming convention (e.g., `component_embedding_newmodel_paramX_valY`).
    *   Example:
        ```python
        class TextEmbeddingNewModelParamXValYComponent(BaseSpecificEmbeddingComponent):
            __tablename__ = "component_embedding_newmodel_paramx_valy"
            # Add any model-specific fields if necessary, though usually not for embeddings
            def __repr__(self):
                return f"TextEmbeddingNewModelParamXValYComponent(id={self.id}, {super().__repr_base__()})"
        ```
2.  **Register the Component**:
    *   Add an entry to the `EMBEDDING_MODEL_REGISTRY` in `dam/models/semantic/text_embedding_component.py`.
    *   Provide the user-facing `model_name` string, the new component class, and any default/defining parameters.
    ```python
    "new-model-paramX": { # User-facing name
        "model_class": TextEmbeddingNewModelParamXValYComponent,
        "default_params": {"paramX": "valY", "dimensions": 256}, # Example params
    }
    ```
3.  **Update `dam/models/semantic/__init__.py`**:
    *   Ensure the new component class is exported in the `__all__` list.
4.  **Database Migration (Alembic)**:
    *   Generate a new Alembic migration: `alembic revision -m "create_table_for_new_model_paramx_valy_embedding"`
    *   In the generated migration script, use SQLAlchemy metadata or `op.create_table()` to define the new table schema based on your component class.
    *   Ensure `op.drop_table()` is present in the `downgrade()` function.
    *   *(Note: Alembic step was deferred in the current task, but is standard procedure).*
5.  **Update Tests**:
    *   Add tests in `tests/test_semantic_functions.py` to verify that the new embedding type works correctly for creation, retrieval, and similarity search. This includes mocking the new model if necessary and ensuring data goes into the new table.

### 2.6. Audio Embedding Components

Similar to text embeddings, the system supports storing and searching audio embeddings.

-   **Base Class**: `dam.models.semantic.audio_embedding_component.BaseSpecificAudioEmbeddingComponent`.
    *   This class includes `embedding_vector` (bytes) and `model_name` (string) columns. The `model_name` column explicitly stores the name of the audio model that generated the embedding (e.g., "vggish", "panns_cnn14").
-   **Specific Component Classes**: For each audio model (and potentially significant parameter variations), a specific component class inheriting from `BaseSpecificAudioEmbeddingComponent` is created.
    *   Example: `AudioEmbeddingVggishDim128Component` (table: `component_audio_embedding_vggish_d128`).
-   **Registry**: `AUDIO_EMBEDDING_MODEL_REGISTRY` in `dam.models.semantic.audio_embedding_component.py` maps model names (e.g., "vggish") to their respective component classes and default parameters.
    ```python
    AUDIO_EMBEDDING_MODEL_REGISTRY: Dict[str, AudioEmbeddingModelInfo] = {
        "vggish": {
            "model_class": AudioEmbeddingVggishDim128Component,
            "default_params": {"dimensions": 128},
        },
        # ... other audio models
    }
    ```
-   **Functions Layer (`audio_functions`)**:
    *   Located in `dam_media_audio.functions.audio_functions.py`.
    *   Responsible for generating, storing, and searching audio embeddings.
    *   Uses `get_audio_embedding_component_class` (which consults the `AUDIO_EMBEDDING_MODEL_REGISTRY`) to determine the correct component class for database operations.
    *   Relies on the `ModelExecutionManager` (see Section 4) for loading audio models (currently mocked).
    *   Key functions:
        *   `generate_audio_embedding_for_entity`: Creates/updates an audio embedding for an entity.
        *   `find_similar_entities_by_audio_embedding`: Searches for entities with similar audio.
-   **System (`AudioProcessingSystem`)**:
    *   A system can be designed to react to `AudioComponent` being added to an entity, or it can be triggered by a command.
    *   Uses `audio_functions` to generate embeddings.
    *   Relies on `dam_fs.functions.file_operations.get_file_path_for_entity` to locate the audio file for an entity.
-   **Commands**:
    *   `AudioSearchCommand` (in `dam_media_audio.commands`): Used to request an audio similarity search.
    *   Handled by `handle_audio_search_command` in `dam_semantic.systems`, which calls the `audio_functions`.

#### Adding a New Audio Embedding Model

1.  **Define Component Class**: In `dam/models/semantic/audio_embedding_component.py`, create a new class inheriting from `BaseSpecificAudioEmbeddingComponent`. Define `__tablename__`.
2.  **Register Component**: Add an entry to `AUDIO_EMBEDDING_MODEL_REGISTRY`.
3.  **Update `dam/models/semantic/__init__.py`**: Export the new component.
4.  **Implement Model Loading**:
    *   Provide a loader function for the actual audio model.
    *   Register this loader with the `ModelExecutionManager` in `audio_functions` (similar to how `MockAudioModel` is handled, but with a real model). The identifier used with `ModelExecutionManager` should be distinct (e.g., `MOCK_AUDIO_MODEL_IDENTIFIER` or a new one like `VGGISH_AUDIO_MODEL_IDENTIFIER`).
5.  **Database Migration (Alembic)**: Create a migration for the new table if specific tables per model are used, or ensure the generic table approach is configured. (Current `BaseSpecificAudioEmbeddingComponent` design with `model_name` column supports a more generic table if desired, but specific tables are also an option per model type).
6.  **Update Tests**: Add tests for `audio_functions` and `AudioProcessingSystem` for the new model.

### 2.7. Tagging Components and Functions

The system includes functionality for both manual and AI-driven (model-generated) tagging of entities.

-   **`TagConceptComponent`** (`dam.models.tags.TagConceptComponent`):
    *   Defines a canonical tag (e.g., "landscape", "character:Alice"). Each unique tag string is stored once.
    *   Attached to an `Entity` that represents the tag itself.
-   **`EntityTagLinkComponent`** (`dam.models.tags.EntityTagLinkComponent`):
    *   Links an entity to a `TagConceptComponent`'s entity, representing a **manually applied tag**.
    *   Can optionally store a `tag_value`.
-   **`ModelGeneratedTagLinkComponent`** (`dam.models.tags.ModelGeneratedTagLinkComponent`):
    *   Links an entity to a `TagConceptComponent`'s entity for tags **generated by an AI model**.
    *   Stores `source_model_name` (e.g., "wd-v1-4-moat-tagger-v2") and `confidence` for the tag.
    *   Model parameters used for generation are defined in a code registry (`TAGGING_MODEL_REGISTRY`) and not stored per link to save space.
    *   Table: `component_model_generated_tag_link`.
-   **`tagging_functions.py`** (`dam_app.functions.tagging_functions.py`):
    *   Manages AI-driven tagging.
    *   Relies on `ModelExecutionManager` (see Section 4) for loading tagging models (currently mocked using `MockWd14Tagger`).
    *   Uses `TAGGING_MODEL_CONCEPTUAL_PARAMS` (in `dam_app.functions.tagging_functions.py`) to store behavioral parameters for specific models (e.g., "wd-v1-4-moat-tagger-v2").
    *   `update_entity_model_tags()`: Generates tags for an entity's image using a specified model and stores them as `ModelGeneratedTagLinkComponent` instances. It typically replaces previous tags from the same model for that entity.
    *   Uses `tag_functions.get_or_create_tag_concept()` to interact with `TagConceptComponent`.
-   **`tag_functions.py`**:
    *   Manages `TagConceptComponent` (creation, retrieval) and manual `EntityTagLinkComponent`.
    *   Includes `get_or_create_tag_concept()` for use by other function modules.
-   **`AutoTaggingSystem`** (`dam_app/systems/auto_tagging_system.py`):
    *   An example system that can be triggered by a command (`AutoTagEntityCommand`).
    *   Uses `tagging_functions` to apply tags from a configured model.

#### Adding a New Auto-Tagging Model

1.  **Define Conceptual Parameters**:
    *   Add an entry to `TAGGING_MODEL_CONCEPTUAL_PARAMS` in `dam_app.functions.tagging_functions.py` for the new model name. Include `default_conceptual_params` (e.g., confidence thresholds, tag limits for prediction) and optionally `model_load_params` if the loader needs specific arguments.
2.  **Implement Loader Function**:
    *   Create a synchronous function that takes `model_name_or_path` (which will be your new model's name/identifier) and an optional `params` dictionary, and returns the loaded model instance.
3.  **Register Loader in `tagging_functions`**:
    *   In `dam_app.functions.tagging_functions.py`, within `get_tagging_model()`, ensure the new loader function is registered with the `ModelExecutionManager` using a suitable `TAGGING_MODEL_IDENTIFIER` (can be generic like "image_tagger" or model-specific if behaviors differ greatly).
    *   Example: `model_manager.register_model_loader(TAGGING_MODEL_IDENTIFIER, _my_new_tagger_loader_sync)`
4.  **Update Systems (Optional)**:
    *   If the new model should be triggered by specific conditions or markers, update or create relevant systems. The `AutoTaggingSystem` can be configured to use different model names.
5.  **Update Tests**:
    *   Add tests for the `tagging_functions` using the new model (likely with mocked model output).
    *   Test any new system logic.

### 2.8. Constructor Expectations
- Components inherit `kw_only=True` constructor behavior from `dam.models.core.base_class.Base`.
- In `BaseComponent`, both the `entity_id` field (linking to an `Entity`) and the `entity` relationship attribute itself are defined with `init=False`. This means they are not set via the component's constructor when you first create an instance of a component.
- Instead, components are instantiated with their own specific data fields (those that are `init=True` by default or explicitly in the component's definition).
- The association with an `Entity` (i.e., setting the `entity_id` and linking the `entity` relationship) is typically handled by the `dam.functions.ecs_functions.add_component_to_entity(session, entity_id, component_instance)` function. This function is called *after* the component instance has been created with its own data.
- Example:
  ```python
  # Create the component with its specific data
  my_component = MyComponent(custom_field="value")
  # Then, associate it with an entity using the functions module
  # actual_entity_id = some_entity.id
  # ecs_functions.add_component_to_entity(session, actual_entity_id, my_component)
  ```

## 3. Functions, Systems, Commands, and Resources

This section outlines the roles and interactions of Functions, Systems, Commands, and Resources.

### 3.1. Functions

-   **Definition and Purpose**: Function modules encapsulate the primary business logic and operations of the application. They act as an intermediary layer, providing the "how-to" for domain-specific tasks.
-   **Structure**: Function modules should be designed as **stateless modules of functions**. They should produce the same result for the same input parameters and not rely on internal instance state that persists across calls.
-   **Interaction**:
    *   Functions operate on Entities and Components, typically by using the `AsyncSession` passed to them or by calling other fine-grained functions (like `ecs_functions` for direct component manipulation).
    *   If a function needs to interact with a stateful resource (e.g., `ModelExecutionManager` for ML models, `FileStorageResource` for world-specific storage, or future remote API clients), that resource instance **must be passed as an argument** to the function.
    *   Functions should **not** use global accessors like `get_default_world()` or similar service locators to find their dependencies.
    *   They are called by Systems (which act as command or event handlers).

### 3.2. Systems

-   **Definition and Purpose**: Systems contain the application's control flow and orchestration logic. They operate on groups of Entities based on the Components they possess, or react to specific Events or Commands. They decide *what* to do and *when*, but delegate the *how* to function modules or the `EcsTransaction` object.
-   **Structure**: Implemented as asynchronous Python functions (`async def`) decorated with `@dam.core.systems.system`. This single decorator can be used in the following ways:
    *   `@system(on_stage=SystemStage.SOME_STAGE)` for stage-based execution.
    *   `@system(on_event=EventType)` for event-driven execution.
    *   `@system(on_command=CommandType)` for command-driven execution.
    *   `@system` (with no arguments) to simply mark a function as a system without scheduling it.
-   **Dependency Injection**:
    *   Systems declare their dependencies using type hints in their parameters. The `WorldScheduler` injects these dependencies.
    *   Common injectable types for Systems include:
        *   The `Event` or `Command` object that triggered the system.
        *   `EcsTransaction`: The primary mechanism for database interaction. Systems should inject this to perform ECS operations like adding/getting components.
        *   `World`: The `World` instance itself can be injected to access world-level information or methods.
        *   `WorldConfig`: The configuration object for the current world.
        *   Any other resource added to the world's `ResourceManager`, which are injected by their type hint.
        *   `WorldSession`: `Annotated[AsyncSession, "WorldSession"]` - **[DEPRECATED]** Direct access to the session is discouraged. Use `EcsTransaction` instead. This is maintained for backward compatibility during the transition.
    *   Systems are responsible for acquiring necessary resources (e.g., the global `ModelExecutionManager` instance) and passing them as arguments to the functions they call.
-   **Execution**: Managed by the `WorldScheduler` based on stages, events, or dispatched commands. The `World` object manages the transaction boundary (see Section 3.6).
-   **Registration**: Systems are registered with a `World` by plugins. Each plugin is responsible for registering its own systems and specifying how they are triggered (stage, event, or command).
-   **Characteristics**:
    *   Should be stateless. All necessary data comes from injected dependencies or queried entities.
    *   Focus on orchestration and flow control, not direct database writes.

### 3.3. Commands

-   **Definition and Purpose**: Commands are requests for the system to perform a specific action. They represent an imperative instruction, such as "ingest this file" or "find similar images". A command is dispatched with the expectation that it will be handled by one or more systems.
-   **Structure**: Commands are simple data-only classes that inherit from a hierarchy of base classes in `dam.core.commands`. They carry the data necessary to execute the action.
    -   `BaseCommand`: The root of all commands.
    -   `EntityCommand`: A base class for commands that operate on a single entity, identified by `entity_id`.
    -   `AnalysisCommand`: A specialized `EntityCommand` for analysis tasks. It includes a `stream_provider` field for direct data access and provides an `open_stream(world)` async context manager to simplify system logic.
    -   Commands can specify an `execution_strategy` (`SERIAL` or `PARALLEL`) to control how their handler systems are executed.
-   **Dispatching**: Commands are sent to the world using `world.dispatch_command(my_command)`. The result is a `SystemExecutor` object. This object is an async generator that yields system events and also provides helper methods (e.g., `get_all_results()`) to consume the stream and process the results. If a command is dispatched from within an existing transaction (i.e., from another command or event handler), it will participate in that same transaction.
-   **Handling**: Systems that handle commands are decorated with `@system(on_command=MyCommand)`. The system function receives the command object as an argument.

### 3.4. Core Asset Events and Commands

To ensure consistency and promote reuse, the core `dam` package defines a set of fundamental events and commands related to asset lifecycle and operations. Plugins should use these core definitions where applicable, rather than defining their own.

-   **Location**:
    -   Core asset events are defined in `dam.events`.
    -   Core asset commands are in `dam.commands`.
-   **Core Events**:
    -   `AssetCreatedEvent`: Fired when a new asset entity is created.
    -   `AssetUpdatedEvent`: Fired when an asset's data is updated.
    -   `AssetDeletedEvent`: Fired when an asset is deleted.
    -   `AssetReadyForMetadataExtractionEvent`: Fired when a batch of assets is ready for metadata extraction. This is a general-purpose event that various plugins can listen to.
-   **Core Commands**:
    -   `GetAssetStreamCommand`: A command to request a readable, seekable binary stream (`typing.BinaryIO`) for an asset's content. This is the primary way for systems to access the file data of an asset. Different plugins (like `dam_fs` for local files or `dam_archive` for files inside archives) can provide handlers for this command.
    -   `GetAssetMetadataCommand`: A command to retrieve metadata for an asset.
    -   `UpdateAssetMetadataCommand`: A command to update metadata for an asset.

### 3.5. Events vs. Commands

It is important to distinguish between Events and Commands to maintain a clean architecture.

-   **Command**: An instruction to do something. It is sent to a specific destination (the `World`'s command dispatcher) with a clear intent. Usually, only one part of the system dispatches a specific command. A command is often (but not always) handled by a single system. Use a command when you want to explicitly trigger a specific piece of business logic.
    -   *Example*: `IngestFileCommand` is dispatched to tell the system to begin the ingestion process for a specific file.
-   **Event**: A notification that something has happened. It is broadcast to the entire system without knowledge of who, if anyone, is listening. Multiple, unrelated systems can listen for the same event to perform their own independent tasks. Use an event when you want to decouple the producer of the notification from its consumers. If an event is dispatched from within an existing transaction, its handlers will participate in that same transaction.
    -   *Example*: `AssetReadyForMetadataExtractionEvent` is fired after an ingestion process completes. Various metadata plugins (for images, audio, documents, etc.) can listen for this event and then dispatch their own specific metadata extraction *commands*. This decouples the ingestion logic from the specific metadata extraction implementations.

### 3.6. Resources

-   **Definition and Purpose**: Resources are shared objects that provide access to external utilities, manage global or world-specific state, or encapsulate connections to infrastructure.
-   **Types**:
    *   **Global Resources**: Singleton instances shared across all worlds if applicable (e.g., `ModelExecutionManager`).
    *   **World-Specific Resources**: Instances specific to a world, often configured by `WorldConfig` (e.g., `FileStorageResource`).
-   **Lifecycle & Management**:
    *   Managed by the `ResourceManager` (`dam.core.resources.ResourceManager`).
    *   Global resources are typically instantiated once when the application starts.
    *   World-specific resources are typically instantiated once per `World` and added to that world's `ResourceManager`.
-   **Access**: Accessed via dependency injection into Systems (using `Annotated[MyResourceType, "Resource"]`). Systems then pass these resources to functions if needed.

### 3.6. Transaction Management and the `EcsTransaction` Object

A core principle of the framework is to ensure data consistency through atomic transactions, especially when a single action (like a command) triggers a chain of subsequent events and commands.

-   **Transaction Boundary**: The transaction is managed by the `World` object. A database transaction begins when a "top-level" command or event is dispatched via `world.dispatch_command()` or `world.dispatch_event()`. The transaction is committed only after the initial operation and *all* subsequent operations it triggers have completed successfully. If any handler in the chain fails, the entire transaction is rolled back.

-   **The `EcsTransaction` Object**: To facilitate this and to provide a controlled interface to the database, the framework uses a dedicated transaction object.
    -   **Purpose**: The `dam.core.transaction.EcsTransaction` class acts as a single point of contact for all ECS-related database operations within a transaction. It wraps the `AsyncSession` and exposes high-level methods for interacting with entities and components (e.g., `add_component`, `get_entity`).
    -   **Lifecycle**: An `EcsTransaction` instance is created by the `World` at the beginning of a top-level transaction. This same instance is then passed down to all systems and functions that are part of that transaction chain.
    -   **Usage in Systems**: Systems should not interact with the database session directly. Instead, they should inject the `EcsTransaction` object and use its methods. The `EcsTransaction` object provides several helper methods to abstract common database operations:
        -   `await transaction.add_component_to_entity(...)`: Adds a component instance to an entity. Best for non-unique components.
        -   `await transaction.add_or_update_component(...)`: A convenient helper that adds or updates a component. It checks if the component is a `UniqueComponent` and, if so, will update an existing instance instead of causing an error. This is the preferred method for managing unique components.
        -   `await transaction.create_entity()`: Creates a new entity.
        -   If an ID is needed immediately for a subsequent operation within the same transaction, `await transaction.flush()` can be called.
    -   **Commit/Rollback**: The underlying session's `commit()` or `rollback()` method is called automatically by the `World` object at the end of the transaction. Systems and functions should **never** call commit or rollback themselves.

This approach ensures that system logic remains focused on orchestration, while the framework guarantees atomicity and provides a safe, domain-specific API for database interactions.

### 3.7. Progress Reporting Events

For long-running commands, especially those involving I/O or significant computation, it is crucial to provide feedback to the caller. The system uses a family of events for this purpose, defined in `dam.system_events.progress`. These events are not dispatched to the world's main event bus but are instead `yield`ed by the system function and consumed by the command dispatcher (`SystemExecutor`).

-   **`ProgressStarted`**: Yielded once at the beginning of a process.
-   **`ProgressUpdate`**: Yielded one or more times to report incremental progress. It can contain `total` and `current` values (e.g., for bytes processed) and a `message`.
-   **`ProgressCompleted`**: Yielded once at the end of a successful process.
-   **`ProgressError`**: Yielded if an error occurs. It contains the `exception` and an optional `message`.

A system that reports progress will have a return type annotation like `AsyncGenerator[Union[SystemProgressEvent, NewEntityCreatedEvent], None]`, indicating it yields a stream of progress and other system events.

## 4. Testing Guidelines

Effective testing is crucial for maintaining code quality and reliability. Specific testing guidelines, including rules about assertions, are outlined in the `AGENTS.md` file at the root of the repository.

## 6. Further Information

For more detailed examples, including how to add new components, manage database migrations (Alembic), and run tests, please consult the [Developer Guide](developer_guide.md).
