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
from dam.traits import Trait
from dam.traits.identifier import TraitIdentifier, TraitImplementationIdentifier
from dam.traits.traits import TraitImplementation


class ComponentA(BaseComponent):
    """A sample component."""

    __tablename__ = "component_a"


class ComponentB(BaseComponent):
    """Another sample component."""

    __tablename__ = "component_b"


class ReadableTrait(Trait):
    """A sample readable trait."""

    identifier = TraitIdentifier(parts=("readable",))

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
    implementation = TraitImplementation(
        trait=ReadableTrait,
        handlers={ReadableTrait.Read: read_handler},
        name="readable",
        description="A readable trait.",
    )
    trait_manager.register(implementation, ComponentA)
    assert trait_manager.get_implementations_for_components({ComponentA}) == [implementation]


def test_trait_manager_get_implementations_for_multiple_components(world: World):
    """Test that trait implementations for multiple components can be retrieved."""
    trait_manager = world.trait_manager
    implementation = TraitImplementation(
        trait=ReadableTrait,
        handlers={ReadableTrait.Read: read_handler},
        name="readable",
        description="A readable trait.",
    )
    trait_manager.register(implementation, (ComponentA, ComponentB))
    assert trait_manager.get_implementations_for_components({ComponentA, ComponentB}) == [implementation]


def test_trait_manager_get_trait_handlers(world: World):
    """Test that trait handlers can be retrieved."""
    trait_manager = world.trait_manager
    implementation = TraitImplementation(
        trait=ReadableTrait,
        handlers={ReadableTrait.Read: read_handler},
        name="readable",
        description="A readable trait.",
    )
    trait_manager.register(implementation, ComponentA)
    handlers = trait_manager.get_trait_handlers({ComponentA})
    assert handlers[ReadableTrait.Read] == read_handler


def test_trait_manager_get_trait_by_id(world: World):
    """Test that a trait can be retrieved by its identifier."""
    trait_manager = world.trait_manager
    implementation = TraitImplementation(
        trait=ReadableTrait,
        handlers={ReadableTrait.Read: read_handler},
        name="readable",
        description="A readable trait.",
    )
    trait_manager.register(implementation, ComponentA)
    trait = trait_manager.get_trait_by_id("readable")
    assert trait is ReadableTrait


def test_trait_manager_uniqueness(world: World):
    """Test that trait identifiers must be unique."""
    trait_manager = world.trait_manager
    implementation = TraitImplementation(
        trait=ReadableTrait,
        handlers={ReadableTrait.Read: read_handler},
        name="readable",
        description="A readable trait.",
    )
    trait_manager.register(implementation, ComponentA)

    class DuplicateTrait(Trait):
        identifier = TraitIdentifier(parts=("readable",))

    implementation2 = TraitImplementation(
        trait=DuplicateTrait,
        handlers={},
        name="readable",
        description="A readable trait.",
    )
    with pytest.raises(ValueError, match="already registered"):
        trait_manager.register(implementation2, ComponentB)


@pytest.mark.asyncio
async def test_get_available_traits_for_entity(world: World, mocker: MockerFixture):
    """Test that available traits for an entity can be retrieved."""
    db = mocker.AsyncMock(spec=DamDatabase)
    db.get_component_types_for_entity = mocker.AsyncMock(return_value={ComponentA})
    world.add_resource(db, DamDatabase)

    implementation = TraitImplementation(
        trait=ReadableTrait,
        handlers={ReadableTrait.Read: read_handler},
        name="readable",
        description="A readable trait.",
    )
    world.trait_manager.register(implementation, ComponentA)

    traits = await world.get_available_traits_for_entity(1)
    assert len(traits) == 1
    assert isinstance(traits[0], ReadableTrait)


def test_trait_manager_get_implementation_by_id(world: World):
    """Test that a trait implementation can be retrieved by its identifier."""
    trait_manager = world.trait_manager
    implementation = TraitImplementation(
        trait=ReadableTrait,
        handlers={ReadableTrait.Read: read_handler},
        name="readable",
        description="A readable trait.",
    )
    trait_manager.register(implementation, ComponentA)
    retrieved_implementation = trait_manager.get_implementation_by_id(
        TraitImplementationIdentifier.from_string("readable:ComponentA")
    )
    assert retrieved_implementation is implementation


def test_trait_manager_get_implementations_for_trait(world: World):
    """Test that all implementations for a trait can be retrieved."""
    trait_manager = world.trait_manager
    implementation1 = TraitImplementation(
        trait=ReadableTrait,
        handlers={ReadableTrait.Read: read_handler},
        name="readable",
        description="A readable trait.",
    )
    trait_manager.register(implementation1, ComponentA)

    class AnotherReadableTrait(Trait):
        identifier = TraitIdentifier(parts=("another_readable",))

    implementation2 = TraitImplementation(
        trait=AnotherReadableTrait,
        handlers={},
        name="another_readable",
        description="Another readable trait.",
    )
    trait_manager.register(implementation2, ComponentB)
    implementations = trait_manager.get_implementations_for_trait(ReadableTrait)
    assert implementations == [implementation1]
