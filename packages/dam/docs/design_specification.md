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
  - [2.7. Tagging Components and Services](#27-tagging-components-and-services)
  - [2.8. Constructor Expectations](#28-constructor-expectations)
- [3. Systems](#3-systems)
  - [3.1. Definition and Purpose](#31-definition-and-purpose)
  - [3.2. Decorators and Stages](#32-decorators-and-stages)
  - [3.3. Dependency Injection](#33-dependency-injection)
- [4. Testing Guidelines](#4-testing-guidelines)
- [5. Further Information](#5-further-information)

## 1. Core Architecture

The system is built upon the Entity-Component-System (ECS) pattern. This pattern promotes flexibility, modularity, and separation of concerns.
- **Entities**: Unique identifiers for assets or concepts.
- **Components**: Data-only classes describing aspects of entities.
- **Systems**: Logic operating on entities based on their components.
- **Plugins**: A modular, Bevy-like plugin system for extending functionality. Plugins are responsible for registering systems and resources with a `World`.

For a comprehensive understanding of the ECS implementation, refer to the [Developer Guide](developer_guide.md).

### 1.1. Dependency Rule

A critical design principle is that the core `dam` package **must not** depend on any of its plugin packages (e.g., `dam_media_image`, `dam_sire`). This ensures the core remains lean and decoupled.

-   **Core `dam` Package**: Contains the fundamental ECS framework, base components, and core services that have no knowledge of specific media types or external model frameworks.
-   **Plugin Packages**: Extend the core functionality. They can depend on `dam`, but `dam` cannot depend on them.
-   **Application Package (`dam_app`)**: The main application that brings everything together. It depends on `dam` and all the necessary plugins.

If a system or service within the `dam` package needs functionality from a plugin, it must be moved to the appropriate plugin package or to the `dam_app` package if it's application-specific logic.

## 2. Models (Components)

### 2.1. Definition and Purpose
Components are data-only Python classes that store specific attributes or properties of an entity. They should not contain business logic.

### 2.2. Naming Conventions
- **Class Names**: `PascalCase` (e.g., `FilePropertiesComponent`).
- **Convention for Table Names**: Generally `component_[name]` (e.g., `component_file_properties`).
    - For specific embedding components, table names are more descriptive, like `component_embedding_minilm_l6_v2_d384`, indicating the model, version, and key parameters (e.g., dimension).
    - Component class names also reflect this specificity, e.g., `TextEmbeddingAllMiniLML6V2Dim384Component`.

### 2.3. Inheritance
- Most components inherit from `dam.models.core.base_component.BaseComponent`.
- Specific embedding components (e.g., `TextEmbeddingAllMiniLML6V2Dim384Component`) inherit from `dam.models.semantic.text_embedding_component.BaseSpecificEmbeddingComponent`, which itself inherits from `BaseComponent`. `BaseSpecificEmbeddingComponent` is an abstract base class and includes common fields for embeddings like `embedding_vector`, `source_component_name`, and `source_field_name`.

### 2.4. Field Definitions
- Component fields should be defined using SQLAlchemy's `Mapped` type hints and `mapped_column` function.
- Example: `original_filename: Mapped[Optional[str]] = mapped_column(String(1024))`
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
-   **Service Layer**: The `semantic_service.py` uses this registry (`get_embedding_component_class` function) to determine the correct component class (and thus, table) for operations like creating, retrieving, or searching embeddings based on the provided `model_name` and `model_params`.
    The service also supports augmenting input text with tags (manual or model-generated) before generating embeddings. This is controlled by parameters like `include_manual_tags` and `include_model_tags_config` in `update_text_embeddings_for_entity`. The `source_field_name` in the stored embedding component will be suffixed (e.g., `_with_tags`) to indicate this augmentation.

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
    *   Add tests in `tests/test_semantic_service.py` to verify that the new embedding type works correctly for creation, retrieval, and similarity search. This includes mocking the new model if necessary and ensuring data goes into the new table.

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
-   **Service Layer (`AudioService`)**:
    *   Located in `dam_media_audio.services.audio_service.py`.
    *   Responsible for generating, storing, and searching audio embeddings.
    *   Uses `get_audio_embedding_component_class` (which consults the `AUDIO_EMBEDDING_MODEL_REGISTRY`) to determine the correct component class for database operations.
    *   Relies on the `ModelExecutionManager` (see Section 4) for loading audio models (currently mocked).
    *   Key functions:
        *   `generate_audio_embedding_for_entity`: Creates/updates an audio embedding for an entity.
        *   `find_similar_entities_by_audio_embedding`: Searches for entities with similar audio.
-   **System (`AudioProcessingSystem`)**:
    *   `audio_embedding_generation_system` in `dam.systems.audio_systems.py`.
    *   Processes entities marked with `NeedsAudioProcessingMarker`.
    *   Uses `AudioService` to generate embeddings.
    *   Relies on `dam.utils.media_utils.get_file_path_for_entity` to locate the audio file for an entity.
-   **Events**:
    *   `AudioSearchQuery` (in `dam.core.events.py`): Used to request an audio similarity search.
    *   Handled by `handle_audio_search_query` in `dam.systems.semantic_systems.py`, which calls the `AudioService`.

#### Adding a New Audio Embedding Model

1.  **Define Component Class**: In `dam/models/semantic/audio_embedding_component.py`, create a new class inheriting from `BaseSpecificAudioEmbeddingComponent`. Define `__tablename__`.
2.  **Register Component**: Add an entry to `AUDIO_EMBEDDING_MODEL_REGISTRY`.
3.  **Update `dam/models/semantic/__init__.py`**: Export the new component.
4.  **Implement Model Loading**:
    *   Provide a loader function for the actual audio model.
    *   Register this loader with the `ModelExecutionManager` in `AudioService` (similar to how `MockAudioModel` is handled, but with a real model). The identifier used with `ModelExecutionManager` should be distinct (e.g., `MOCK_AUDIO_MODEL_IDENTIFIER` or a new one like `VGGISH_AUDIO_MODEL_IDENTIFIER`).
5.  **Database Migration (Alembic)**: Create a migration for the new table if specific tables per model are used, or ensure the generic table approach is configured. (Current `BaseSpecificAudioEmbeddingComponent` design with `model_name` column supports a more generic table if desired, but specific tables are also an option per model type).
6.  **Update Tests**: Add tests for `AudioService` and `AudioProcessingSystem` for the new model.

### 2.7. Tagging Components and Services

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
-   **`TaggingService`** (`dam_app.services.tagging_service.py`):
    *   Manages AI-driven tagging.
    *   Relies on `ModelExecutionManager` (see Section 4) for loading tagging models (currently mocked using `MockWd14Tagger`).
    *   Uses `TAGGING_MODEL_CONCEPTUAL_PARAMS` (in `dam_app.services.tagging_service.py`) to store behavioral parameters for specific models (e.g., "wd-v1-4-moat-tagger-v2").
    *   `update_entity_model_tags()`: Generates tags for an entity's image using a specified model and stores them as `ModelGeneratedTagLinkComponent` instances. It typically replaces previous tags from the same model for that entity.
    *   Uses `tag_service.get_or_create_tag_concept()` to interact with `TagConceptComponent`.
-   **`tag_service.py`**:
    *   Manages `TagConceptComponent` (creation, retrieval) and manual `EntityTagLinkComponent`.
    *   Includes `get_or_create_tag_concept()` for use by other services.
-   **`AutoTaggingSystem`** (`dam_app/systems/auto_tagging_system.py`):
    *   An example system that processes entities marked with `NeedsAutoTaggingMarker`.
    *   Uses `TaggingService` to apply tags from a configured model.

#### Adding a New Auto-Tagging Model

1.  **Define Conceptual Parameters**:
    *   Add an entry to `TAGGING_MODEL_CONCEPTUAL_PARAMS` in `dam_app.services.tagging_service.py` for the new model name. Include `default_conceptual_params` (e.g., confidence thresholds, tag limits for prediction) and optionally `model_load_params` if the loader needs specific arguments.
2.  **Implement Loader Function**:
    *   Create a synchronous function that takes `model_name_or_path` (which will be your new model's name/identifier) and an optional `params` dictionary, and returns the loaded model instance.
3.  **Register Loader in `TaggingService`**:
    *   In `dam_app.services.tagging_service.py`, within `get_tagging_model()`, ensure the new loader function is registered with the `ModelExecutionManager` using a suitable `TAGGING_MODEL_IDENTIFIER` (can be generic like "image_tagger" or model-specific if behaviors differ greatly).
    *   Example: `model_manager.register_model_loader(TAGGING_MODEL_IDENTIFIER, _my_new_tagger_loader_sync)`
4.  **Update Systems (Optional)**:
    *   If the new model should be triggered by specific conditions or markers, update or create relevant systems. The `AutoTaggingSystem` can be configured to use different model names.
5.  **Update Tests**:
    *   Add tests for the `TaggingService` using the new model (likely with mocked model output).
    *   Test any new system logic.

### 2.8. Constructor Expectations
- Components inherit `kw_only=True` constructor behavior from `dam.models.core.base_class.Base`.
- In `BaseComponent`, both the `entity_id` field (linking to an `Entity`) and the `entity` relationship attribute itself are defined with `init=False`. This means they are not set via the component's constructor when you first create an instance of a component.
- Instead, components are instantiated with their own specific data fields (those that are `init=True` by default or explicitly in the component's definition).
- The association with an `Entity` (i.e., setting the `entity_id` and linking the `entity` relationship) is typically handled by the `dam.services.ecs_service.add_component_to_entity(session, entity_id, component_instance)` function. This function is called *after* the component instance has been created with its own data.
- Example:
  ```python
  # Create the component with its specific data
  my_component = MyComponent(custom_field="value")
  # Then, associate it with an entity using the service
  # actual_entity_id = some_entity.id
  # ecs_service.add_component_to_entity(session, actual_entity_id, my_component)
  ```

## 3. Services, Systems, and Resources

This section outlines the roles and interactions of Services, Systems, and Resources.

### 3.1. Services

-   **Definition and Purpose**: Services encapsulate the primary business logic and operations of the application. They act as an intermediary layer, providing the "how-to" for domain-specific tasks.
-   **Structure**: Services should be designed as **stateless modules of functions**. They should produce the same result for the same input parameters and not rely on internal instance state that persists across calls.
-   **Interaction**:
    *   Services operate on Entities and Components, typically by using the `AsyncSession` passed to them or by calling other fine-grained services (like `ecs_service` for direct component manipulation).
    *   If a Service function needs to interact with a stateful resource (e.g., `ModelExecutionManager` for ML models, `FileStorageResource` for world-specific storage, or future remote API clients), that resource instance **must be passed as an argument** to the service function.
    *   Service functions should **not** use global accessors like `get_default_world()` or similar service locators to find their dependencies.
    *   They are called by Systems.

### 3.2. Systems

-   **Definition and Purpose**: Systems contain the application's control flow and orchestration logic. They operate on groups of Entities based on the Components they possess or react to specific Events occurring within the application. They decide *what* to do and *when*, but delegate the *how* to Services.
-   **Structure**: Implemented as asynchronous Python functions (`async def`) decorated with `@dam.core.systems.system(stage=SystemStage.SOME_STAGE)` for stage-based execution or `@dam.core.systems.listens_for(EventType)` for event-driven execution.
-   **Dependency Injection**:
    *   Systems declare their dependencies using `typing.Annotated` type hints in their parameters. The `WorldScheduler` injects these dependencies.
    *   Common injectable types for Systems include:
        *   `WorldSession`: `Annotated[AsyncSession, "WorldSession"]` - The active SQLAlchemy session for the current world.
        *   `Resource[ResourceType]`: `Annotated[MyResourceType, "Resource"]` - Shared resources. This includes world-specific resources (like `FileStorageResource`) and global resources (like the `ModelExecutionManager`, see Section 4).
        *   `MarkedEntityList[MarkerComponentType]`: `Annotated[List[Entity], "MarkedEntityList", MyMarkerComponent]` - A list of entities from the current world that have the specified marker component.
        *   `WorldContext`: Provides access to `WorldSession`, world name, and `WorldConfig`.
    *   Systems are responsible for acquiring necessary resources (e.g., the global `ModelExecutionManager` instance, or specific model clients vended by it) and passing them as arguments to the service functions they call.
-   **Execution**: Managed by the `WorldScheduler` based on stages or events. The `WorldScheduler` also handles session commits/rollbacks per stage or event cycle.
-   **Registration**: Systems are registered with a `World` by plugins. Each plugin is responsible for registering its own systems.
-   **Characteristics**:
    *   Should be stateless. All necessary data comes from injected dependencies or queried entities.
    *   Focus on orchestration and flow control.

### 3.3. Resources

-   **Definition and Purpose**: Resources are shared objects that provide access to external utilities, manage global or world-specific state, or encapsulate connections to infrastructure.
-   **Types**:
    *   **Global Resources**: Singleton instances shared across all worlds if applicable (e.g., `ModelExecutionManager`).
    *   **World-Specific Resources**: Instances specific to a world, often configured by `WorldConfig` (e.g., `FileStorageResource`).
-   **Lifecycle & Management**:
    *   Managed by the `ResourceManager` (`dam.core.resources.ResourceManager`).
    *   Global resources are typically instantiated once when the application starts.
    *   World-specific resources are typically instantiated once per `World` and added to that world's `ResourceManager`.
-   **Access**: Accessed via dependency injection into Systems (using `Annotated[MyResourceType, "Resource"]`). Systems then pass these resources to service functions if needed.

## 4. Testing Guidelines

Effective testing is crucial for maintaining code quality and reliability. Specific testing guidelines, including rules about assertions, are outlined in the `AGENTS.md` file at the root of the repository.

## 6. Further Information

For more detailed examples, including how to add new components, manage database migrations (Alembic), and run tests, please consult the [Developer Guide](developer_guide.md).
