# Design Specification

This document outlines the design principles and guidelines for developing models, systems, and other architectural components within the ECS Digital Asset Management (DAM) system.

## Table of Contents

- [1. Core Architecture](#1-core-architecture)
- [2. Models (Components)](#2-models-components)
  - [2.1. Definition and Purpose](#21-definition-and-purpose)
  - [2.2. Naming Conventions](#22-naming-conventions)
  - [2.3. Inheritance](#23-inheritance)
  - [2.4. Field Definitions](#24-field-definitions)
  - [2.5. Constructor Expectations](#25-constructor-expectations)
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

For a comprehensive understanding of the ECS implementation, refer to the [Developer Guide](developer_guide.md).

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
- Specific embedding tables do *not* store `model_name` or `model_parameters` as columns; this information is inherent in the choice of table.

### 2.5. Semantic Embedding Components

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

### 2.6. Constructor Expectations
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

## 3. Systems

### 3.1. Definition and Purpose
Systems encapsulate the logic that operates on entities. They typically query for entities possessing a specific combination of components and perform actions based on that data.

### 3.2. Decorators and Stages
- Systems are typically asynchronous Python functions (`async def`).
- They must be decorated with `@dam.core.systems.system(stage=SystemStage.SOME_STAGE)`.
- `SystemStage` (from `dam.core.stages`) defines distinct phases in the application's processing lifecycle, allowing for ordered execution.

### 3.3. Dependency Injection
- Systems declare their dependencies using `typing.Annotated` type hints in their parameters. The `WorldScheduler` injects these dependencies.
- Common injectable types include:
    - `WorldSession`: `Annotated[AsyncSession, "WorldSession"]` - The active SQLAlchemy session.
    - `Resource[ResourceType]`: `Annotated[MyResourceType, "Resource"]` - Shared resources like file operators.
    - `MarkedEntityList[MarkerComponentType]`: `Annotated[List[Entity], "MarkedEntityList", MyMarkerComponent]` - A list of entities that have the specified marker component.
    - `WorldContext`: Provides access to session, world name, and config.

## 4. Testing Guidelines

Effective testing is crucial for maintaining code quality and reliability. Specific testing guidelines, including rules about assertions, are outlined in the `AGENTS.md` file at the root of the repository.

## 5. Further Information

For more detailed examples, including how to add new components, manage database migrations (Alembic), and run tests, please consult the [Developer Guide](developer_guide.md).
