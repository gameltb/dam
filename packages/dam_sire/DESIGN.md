# `dam_sire` Package Design

This document outlines the design for the `dam_sire` package, which will replace the functionality of the removed `ModelExecutionManager`.

## 1. Overview

The `dam_sire` package will act as a plugin for the `dam` application, integrating the `sire` model execution framework. This will provide a more robust and flexible way to manage and run machine learning models within the DAM system.

## 2. Key Components

### 2.1. `SireResource`

A new resource class, `SireResource`, will be created. This resource will be the main interface for interacting with the `sire` framework.

-   **Purpose:** To abstract the details of the `sire` framework and provide a simple API for loading and running models.
-   **Implementation:** It will be a wrapper around `sire`'s core functionalities, such as `sire.core.runtime_resource_management.RuntimeResourceManager`.
-   **Lifecycle:** A single instance of `SireResource` will be created and added to each `dam` world.

### 2.2. `SirePlugin`

A new plugin class, `SirePlugin`, will be responsible for setting up the `sire` integration.

-   **Purpose:** To initialize and register the `SireResource` in the `dam` world.
-   **Implementation:** It will implement the `dam.core.plugin.Plugin` protocol. In its `build` method, it will instantiate the `SireResource` and add it to the world's resource manager.

## 3. Integration with `dam`

The `dam_sire` package will be integrated into the `dam` application as follows:

1.  **Plugin Registration:** The `dam_app` will be responsible for loading the `SirePlugin`. This will be done in `packages/dam_app/src/dam_app/cli.py`, similar to how other plugins are loaded.
2.  **Service Usage:** The services that require model execution (e.g., `audio_service`, `tagging_service`) will be updated to use the `SireResource`. They will get the resource from the world via dependency injection in the systems that call them.

## 4. Example Usage

```python
# In a system in dam_semantic

from dam_sire.resource import SireResource

@system(...)
async def my_system(sire_resource: Annotated[SireResource, "Resource"]):
    # The system gets the SireResource via dependency injection

    # The system then calls a service, passing the resource
    await my_service.do_something_with_a_model(sire_resource, ...)

# In a service in dam_semantic

async def do_something_with_a_model(sire_resource: SireResource, ...):
    # The service uses the SireResource to get a model and run it
    model = await sire_resource.get_model(...)
    result = await model.predict(...)
```

This design provides a clean separation of concerns, where the `dam_sire` package encapsulates all the logic related to the `sire` framework, and the other parts of the application can use it through a simple and well-defined API.
