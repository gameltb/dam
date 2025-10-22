"""Unit tests for the Trait system."""

from dataclasses import dataclass
from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from dam.commands.core import EntityCommand
from dam.core.database import DatabaseManager as DamDatabase
from dam.core.types import FileStreamProvider, StreamProvider
from dam.core.world import World
from dam.models import BaseComponent
from dam.system_events.base import BaseSystemEvent
from dam.traits import Trait, TraitImplementation


class ComponentA(BaseComponent):
    """A sample component."""

    __tablename__ = "component_a"


class ComponentB(BaseComponent):
    """Another sample component."""

    __tablename__ = "component_b"


class ReadableTrait(Trait):
    """A sample readable trait."""

    name = "readable"

    @dataclass
    class Read(EntityCommand[StreamProvider, BaseSystemEvent]):
        """A sample read command."""

        pass


async def read_handler(_cmd: ReadableTrait.Read, _world: World) -> StreamProvider:
    """Return a sample read handler."""
    return FileStreamProvider(Path("test"))


@pytest.fixture
def world() -> World:
    """Return a World instance."""
    return World("test")


def test_trait_manager_register(world: World):
    """Test that a trait implementation can be registered."""
    trait_manager = world.trait_manager
    implementation = TraitImplementation(trait=ReadableTrait, handlers={ReadableTrait.Read: read_handler})
    trait_manager.register(ComponentA, implementation)
    assert trait_manager.get_implementations_for_components({ComponentA}) == [implementation]


def test_trait_manager_get_implementations_for_multiple_components(world: World):
    """Test that trait implementations for multiple components can be retrieved."""
    trait_manager = world.trait_manager
    implementation = TraitImplementation(trait=ReadableTrait, handlers={ReadableTrait.Read: read_handler})
    trait_manager.register((ComponentA, ComponentB), implementation)
    assert trait_manager.get_implementations_for_components({ComponentA, ComponentB}) == [implementation]


def test_trait_manager_get_trait_handlers(world: World):
    """Test that trait handlers can be retrieved."""
    trait_manager = world.trait_manager
    implementation = TraitImplementation(trait=ReadableTrait, handlers={ReadableTrait.Read: read_handler})
    trait_manager.register(ComponentA, implementation)
    handlers = trait_manager.get_trait_handlers({ComponentA})
    assert handlers[ReadableTrait.Read] == read_handler


@pytest.mark.asyncio
async def test_get_available_traits_for_entity(world: World, mocker: MockerFixture):
    """Test that available traits for an entity can be retrieved."""
    db = mocker.AsyncMock(spec=DamDatabase)
    db.get_component_types_for_entity = mocker.AsyncMock(return_value={ComponentA})
    world.add_resource(db, DamDatabase)

    implementation = TraitImplementation(trait=ReadableTrait, handlers={ReadableTrait.Read: read_handler})
    world.trait_manager.register(ComponentA, implementation)

    traits = await world.get_available_traits_for_entity(1)
    assert len(traits) == 1
    assert isinstance(traits[0], ReadableTrait)
