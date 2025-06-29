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
- **Convention for Table Names**: `component_[name]` (e.g., `component_file_properties`). While not enforced by `BaseComponent` itself, this is the recommended naming convention.

### 2.3. Inheritance
- All components must inherit from `dam.models.core.base_component.BaseComponent`. This base class provides common fields (`id`, `entity_id`, `created_at`, `updated_at`) and SQLAlchemy integration.

### 2.4. Field Definitions
- Component fields should be defined using SQLAlchemy's `Mapped` type hints and `mapped_column` function.
- Example: `original_filename: Mapped[Optional[str]] = mapped_column(String(1024))`

### 2.5. Constructor Expectations
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
