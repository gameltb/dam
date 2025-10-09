"""Core World class and global world registry for the DAM system."""

import logging
from collections.abc import Callable
from typing import Any, TypeVar, cast

from dam.commands.core import BaseCommand, EventType, ResultType
from dam.contexts import ContextProvider
from dam.core.config import WorldConfig
from dam.core.executor import SystemExecutor
from dam.core.operations import AssetOperation
from dam.core.plugin import Plugin
from dam.core.resources import ResourceManager
from dam.core.stages import SystemStage
from dam.core.systems import WorldScheduler
from dam.events import BaseEvent
from dam.system_events.base import SystemResultEvent

logger = logging.getLogger(__name__)

# Type variable for generic resource types
T = TypeVar("T")


class World:
    """
    Represents an isolated ECS (Entity Component System) world.

    Each World encapsulates its own configuration, database connection,
    resource manager, and system scheduler. This allows for multiple
    independent worlds to coexist within the same application, each with
    potentially different settings, data stores, and behaviors.
    """

    def __init__(self, world_config: WorldConfig):
        """Initialize the World."""
        if not isinstance(world_config, WorldConfig):
            raise TypeError(f"world_config must be an instance of WorldConfig, got {type(world_config)}")

        self.name: str = world_config.name
        self.config: WorldConfig = world_config
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
        self.logger.info("Creating minimal World instance: %s", self.name)

        self.resource_manager: ResourceManager = ResourceManager()
        self.scheduler: WorldScheduler = WorldScheduler(world=self)
        self._registered_plugin_types: set[type[Plugin]] = set()
        self.asset_operations: dict[str, AssetOperation] = {}
        self.context_providers: dict[type[Any], ContextProvider[Any]] = {}
        self.add_resource(self)
        self.logger.info("Minimal World '%s' instance created. Base resources to be populated externally.", self.name)

    def register_asset_operation(self, operation: AssetOperation) -> None:
        """Register an asset operation with the world."""
        if operation.name in self.asset_operations:
            self.logger.warning("Overwriting asset operation for name %s.", operation.name)
        self.asset_operations[operation.name] = operation
        self.logger.info("Asset operation '%s' registered in world '%s'.", operation.name, self.name)

    def get_asset_operation(self, name: str) -> AssetOperation | None:
        """Retrieve a registered asset operation by its name."""
        return self.asset_operations.get(name)

    def get_all_asset_operations(self) -> list[AssetOperation]:
        """Return a list of all registered asset operations."""
        return list(self.asset_operations.values())

    def add_resource(self, instance: object, resource_type: type[Any] | None = None) -> None:
        """Add a resource to the world's resource manager."""
        self.resource_manager.add_resource(instance, resource_type)
        self.logger.debug("Added resource type %s to World '%s'.", resource_type or type(instance), self.name)

    def get_resource(self, resource_type: type[T]) -> T:
        """Get a resource from the world's resource manager."""
        return self.resource_manager.get_resource(resource_type)

    def has_resource(self, resource_type: type[Any]) -> bool:
        """Check if a resource is available in the world's resource manager."""
        return self.resource_manager.has_resource(resource_type)

    def register_context_provider(self, type_hint: type[Any], provider: ContextProvider[Any]) -> None:
        """Register a context provider for a given type hint."""
        if type_hint in self.context_providers:
            self.logger.warning("Overwriting context provider for type %s.", type_hint)
        self.context_providers[type_hint] = provider

    def get_context(self, context_type: type[T]) -> ContextProvider[T]:
        """Get a context provider for a given type hint."""
        provider = self.context_providers.get(context_type)
        if provider is None:
            raise KeyError(f"No context provider registered for type {context_type}")
        return provider

    def has_context(self, context_type: type[Any]) -> bool:
        """Check if a context provider is registered for a given type hint."""
        return context_type in self.context_providers

    def add_plugin(self, plugin: Plugin) -> "World":
        """Add a plugin to the world."""
        plugin_type = type(plugin)
        if plugin_type not in self._registered_plugin_types:
            self.logger.info("Adding plugin %s to world '%s'.", plugin_type.__name__, self.name)
            plugin.build(self)
            self._registered_plugin_types.add(plugin_type)
        else:
            self.logger.debug(
                "Plugin %s is already registered in world '%s'. Skipping.", plugin_type.__name__, self.name
            )
        return self

    def register_system(
        self,
        system_func: Callable[..., Any],
        stage: SystemStage | None = None,
        event_type: type[BaseEvent] | None = None,
        command_type: type[BaseCommand[Any, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        """Register a system with the world's scheduler."""
        num_triggers = sum(1 for trigger in [stage, event_type, command_type] if trigger is not None)
        if num_triggers > 1:
            raise ValueError("A system can only be registered for one trigger type (stage, event, or command).")

        self.scheduler.register_system_for_world(
            system_func, stage=stage, event_type=event_type, command_type=command_type, **kwargs
        )

        if stage:
            self.logger.info(
                "System %s registered for stage %s in world '%s'.", system_func.__name__, stage.name, self.name
            )
        elif event_type:
            self.logger.info(
                "System %s registered for event %s in world '%s'.", system_func.__name__, event_type.__name__, self.name
            )
        elif command_type:
            self.logger.info(
                "System %s registered for command %s in world '%s'.",
                system_func.__name__,
                command_type.__name__,
                self.name,
            )

    async def execute_stage(self, stage: SystemStage) -> None:
        """Execute all systems registered for a given stage."""
        self.logger.info("Executing stage '%s' for World '%s'.", stage.name, self.name)
        await self.scheduler.execute_stage(stage)

    async def dispatch_event(self, event: BaseEvent) -> None:
        """Dispatch an event to all registered handlers."""
        self.logger.info("Dispatching event '%s' for World '%s'.", type(event).__name__, self.name)
        await self.scheduler.dispatch_event(event)

    def dispatch_command(
        self, command: BaseCommand[ResultType, EventType], **kwargs: Any
    ) -> SystemExecutor[ResultType, EventType]:
        """Dispatch a command to its registered handlers."""
        self.logger.info("Dispatching command '%s' for World '%s'.", type(command).__name__, self.name)
        return self.scheduler.dispatch_command(command, **kwargs)

    async def execute_one_time_system(self, system_func: Callable[..., Any], **kwargs: Any) -> Any:
        """
        Execute a single, dynamically provided system function immediately.

        Manages session creation and closure if an external session is not provided.
        Returns the result of the system function.
        """
        self.logger.info(
            "Executing one-time system '%s' for World '%s' with kwargs: %s.",
            system_func.__name__,
            self.name,
            kwargs,
        )

        executor = self.scheduler.execute_one_time_system(system_func, **kwargs)
        async for event in executor:
            if isinstance(event, SystemResultEvent):
                return cast(SystemResultEvent[Any], event).result
        return None

    def __repr__(self) -> str:
        """Return a string representation of the World."""
        return f"<World name='{self.name}' config='{self.config!r}'>"


# This space is intentionally left blank.
