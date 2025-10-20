# DAM 系统测试方法论

## 1. 概述

本文档为 DAM 系统及其插件的开发者提供了一套标准的“系统测试”方法。这里的“系统测试”指的是测试一个或多个系统（System）在一个受控的、隔离的世界（World）实例中的行为和交互。

此方法论旨在取代之前依赖于大型、共享、隐式 fixture（如 `test_world_alpha`）的测试方式，以解决由此带来的测试间依赖、脆弱性和难以调试的问题。新的方法利用了 DAM 核心的动态世界实例化功能，使测试更加健壮、明确和易于维护。

## 2. 核心原则

1.  **测试隔离 (Isolation)**：每个测试都应在自己的、完全独立的 `World` 实例中运行。这意味着每个测试都有自己的内存状态和自己的临时数据库，测试之间绝不共享状态。

2.  **明确声明依赖 (Explicitness)**：每个测试必须明确声明它所依赖的插件。这是通过在测试设置阶段提供这些插件对应的 `ConfigComponent` 实例来实现的。测试环境中不会再有“默认”加载或自动加载的插件。

3.  **可复用工厂 (Reusable Factories)**：世界的创建和销毁逻辑（包括数据库的建立和拆除）是通用的。我们将通过一个可复用的 Pytest Fixture 工厂来封装这些逻辑，避免在每个测试中重复编写设置（setup）和拆除（teardown）代码。

## 3. 核心工具：`world_factory` Fixture

这套测试方法的核心是一个名为 `world_factory` 的 Pytest Fixture。这个 fixture **不是**一个 `World` 实例，而是一个**能够创建 `World` 实例的异步函数（工厂）**。

### 3.1. 功能

`world_factory` 接收一个世界名称和一系列 `ConfigComponent` 实例，然后返回一个功能齐全、完全隔离的 `World` 实例。它会在后台处理：
*   为新世界创建一个独立的临时数据库。
*   调用核心的 `create_world_from_components` 函数。
*   在测试结束后，自动清理数据库和注销世界。

### 3.2. 实现示例

这个 fixture 应该被定义在共享的测试工具包中（例如 `packages/dam_test_utils/src/dam_test_utils/fixtures.py`），以便所有插件的测试都可以使用它。

```python
# In dam_test_utils/fixtures.py

import pytest_asyncio
from dam.core.world import World
from dam.core.world_manager import create_world_from_components, world_manager
from dam.models.config import ConfigComponent
from dam.plugins.core import CoreSettingsComponent

@pytest_asyncio.fixture
async def world_factory(test_db_factory, tmp_path_factory):
    """
    Pytest fixture that provides a factory for creating isolated World instances.
    """
    created_worlds = []

    async def _create_world(world_name: str, components: list[ConfigComponent]) -> World:
        # Ensure a core settings component with a unique DB is present.
        if not any(isinstance(c, CoreSettingsComponent) for c in components):
            temp_db_url = await test_db_factory(f"db_{world_name}")
            temp_alembic_path = tmp_path_factory.mktemp(f"alembic_{world_name}")
            components.append(
                CoreSettingsComponent(
                    plugin_name="core",
                    database_url=temp_db_url,
                    alembic_path=str(temp_alembic_path),
                )
            )

        # Create the world using the core factory function
        world = create_world_from_components(world_name, components)
        created_worlds.append(world.name)
        return world

    yield _create_world

    # Teardown: Unregister all worlds created by this factory
    for name in created_worlds:
        world_manager.unregister_world(name)
```

## 4. 编写系统测试的步骤

使用 `world_factory`，编写一个系统测试遵循清晰的 **Arrange-Act-Assert** 模式。

### 示例场景

我们将测试 `dam-fs` 插件中的 `add_file_properties_handler` 系统。这个系统处理 `AddFilePropertiesCommand` 命令，为一个实体添加 `FileLocationComponent` 和 `FilenameComponent`。

### 测试代码

```python
# In packages/dam_fs/tests/test_asset_lifecycle_systems.py

import pytest
from pathlib import Path
from dam.core.world import World
from dam_fs.commands import AddFilePropertiesCommand
from dam_fs.models import FileLocationComponent, FilenameComponent
from dam_fs.settings import FsSettingsComponent

@pytest.mark.asyncio
async def test_add_file_properties_system(world_factory):
    # 1. Arrange: 创建一个只包含 core 和 dam-fs 插件的世界

    # 1a. 定义这个世界需要的配置组件
    test_asset_path = Path("/tmp/test_assets")
    test_asset_path.mkdir(exist_ok=True)
    fs_settings = FsSettingsComponent(
        plugin_name="dam-fs",
        asset_storage_path=str(test_asset_path)
    )

    # 1b. 使用工厂创建世界
    # CoreSettingsComponent 会被工厂自动添加
    test_world: World = await world_factory("fs_test_world", [fs_settings])

    # 1c. 准备初始状态：创建一个实体
    async with test_world.get_transaction() as tx:
        my_entity = await tx.create_entity()
        await tx.flush()

    # 2. Act: 在创建的世界中分派命令
    command = AddFilePropertiesCommand(
        entity_id=my_entity.id,
        url="file:///tmp/my_asset.txt",
        filename="my_asset.txt"
    )
    # The system we are testing is the handler for this command
    await test_world.execute_command(command).get_all_results()


    # 3. Assert: 验证世界状态的改变
    async with test_world.get_transaction() as tx:
        # 检查 FileLocationComponent 是否被正确添加
        file_loc = await tx.get_component(my_entity.id, FileLocationComponent)
        assert file_loc is not None
        assert file_loc.url == "file:///tmp/my_asset.txt"

        # 检查 FilenameComponent 是否被正确添加
        filename_comp = await tx.get_component(my_entity.id, FilenameComponent)
        assert filename_comp is not None
        assert filename_comp.filename == "my_asset.txt"

```

## 5. 优势

*   **健壮性**：测试之间完全隔离，消除了因共享状态导致的意外失败。
*   **清晰性**：每个测试都明确声明了其依赖的插件和配置，使得测试的意图一目了然。
*   **可维护性**：当插件的配置需求改变时，只需要更新创建相应 `ConfigComponent` 的代码，而不需要修改底层的 fixture logic。
*   **灵活性**：可以轻松地为任何插件组合创建测试世界，从而能够方便地测试插件之间的复杂交互。
