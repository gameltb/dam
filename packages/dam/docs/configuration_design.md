# Plugin Configuration Design

This document outlines the design for the component-based plugin configuration system in the DAM.

## 1. Core Concepts

The configuration for each plugin is managed through two main classes: a `SettingsModel` and a `ConfigComponent`.

-   **`SettingsModel`**: A Pydantic `BaseModel` that defines the structure and validation rules for a plugin's configuration as it appears in the `dam.toml` file. It should only contain the raw configuration fields.

-   **`ConfigComponent`**: A `dam.models.config.ConfigComponent` subclass that holds the validated and processed configuration for a plugin within a `World`. This component is what systems and resources interact with at runtime. It can derive its fields from the `SettingsModel` and may include additional runtime-computed values.

## 2. Configuration Loading Flow

1.  **TOML Parsing**: The `dam.core.config_loader` reads the `dam.toml` file.
2.  **Plugin Discovery**: The loader finds all available plugins.
3.  **Settings Validation**: For each world and plugin defined in the TOML file, the loader uses the plugin's registered `SettingsModel` to validate the corresponding settings section. It also handles environment variable overrides at this stage.
4.  **Component Creation**: The loader then instantiates the plugin's registered `ConfigComponent`, populating it with the validated settings.
5.  **Resource Injection**: The created `ConfigComponent` instance is added as a resource to the `World`.

## 3. How to Add Configuration to a Plugin

To add configuration to a new or existing plugin, follow these steps:

### Step 1: Define the `SettingsModel`

In your plugin's `settings.py` file, create a class that inherits from `dam.models.config.SettingsModel`. Define the configuration fields using Pydantic field types.

**Example**:
```python
# packages/my_plugin/settings.py
from dam.models.config import SettingsModel

class MyPluginSettingsModel(SettingsModel):
    """Defines the configuration fields for MyPlugin."""
    some_api_key: str
    timeout_seconds: int = 60
```

### Step 2: Define the `ConfigComponent`

In the same `settings.py` file, create a class that inherits from `dam.models.config.ConfigComponent`. This class should mirror the fields from your `SettingsModel`.

**Field Naming Convention**: All fields in the `ConfigComponent` should be **lowercase**.

**Example**:
```python
# packages/my_plugin/settings.py
from dam.models.config import ConfigComponent

class MyPluginSettingsComponent(ConfigComponent):
    """Component holding the live configuration for MyPlugin."""
    some_api_key: str
    timeout_seconds: int
```

### Step 3: Register in the Plugin

In your plugin's main `plugin.py` file, attach the `SettingsModel` and `ConfigComponent` to your `Plugin` class.

**Example**:
```python
# packages/my_plugin/plugin.py
from dam.core.plugin import Plugin
from .settings import MyPluginSettingsModel, MyPluginSettingsComponent

class MyPlugin(Plugin):
    Settings = MyPluginSettingsModel
    SettingsComponent = MyPluginSettingsComponent

    def build(self, world: "World") -> None:
        # Plugin build logic...
        pass
```

### Step 4: Accessing Configuration

Systems and resources within the plugin can now access the configuration by requesting the `ConfigComponent` from the `World`'s resources.

**Example (in a System)**:
```python
from .settings import MyPluginSettingsComponent

@system(on_command=MyCommand)
async def my_system(cmd: MyCommand, world: World):
    settings = world.get_resource(MyPluginSettingsComponent)
    api_key = settings.some_api_key
    # ... use the settings
```

**Example (in a Resource)**:
```python
from .settings import MyPluginSettingsComponent

class MyResource:
    def __init__(self, settings: MyPluginSettingsComponent):
        self.timeout = settings.timeout_seconds
        # ...
```

This design ensures that configuration is type-safe, validated, and easily accessible throughout the plugin in a decoupled manner.
