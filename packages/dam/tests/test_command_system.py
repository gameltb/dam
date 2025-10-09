"""Tests for the command system."""

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Annotated

import pytest

from dam.commands.core import BaseCommand
from dam.core.systems import system
from dam.core.world import World
from dam.system_events.base import BaseSystemEvent


# Define some test commands
@dataclass
class SimpleCommand(BaseCommand[str, BaseSystemEvent]):
    """A simple command that returns a string."""

    data: str


@dataclass
class AnotherCommand(BaseCommand[int, BaseSystemEvent]):
    """Another simple command that returns an integer."""

    value: int


@dataclass
class FailingCommand(BaseCommand[None, BaseSystemEvent]):
    """A command that is expected to fail."""

    pass


@dataclass
class ListCommand(BaseCommand[list[str], BaseSystemEvent]):
    """A command that returns a list of strings."""

    pass


@dataclass
class MismatchCommand(BaseCommand[int, BaseSystemEvent]):
    """A command that has a handler with a mismatched return type."""

    pass


@dataclass
class CustomEvent(BaseSystemEvent):
    """A custom event for testing."""

    message: str


@dataclass
class StreamingCommand(BaseCommand[None, CustomEvent]):
    """A command that streams custom events."""

    pass


@dataclass
class LegacyCommand(BaseCommand[str, BaseSystemEvent]):
    """A command for testing backward compatibility."""

    pass


# Define some test systems (command handlers)
@system(on_command=SimpleCommand)
def simple_handler_one(cmd: SimpleCommand) -> str:
    """Handle a simple command."""
    return f"Handled one: {cmd.data}"


@system(on_command=SimpleCommand)
async def simple_handler_two(cmd: SimpleCommand) -> str:
    """Handle a simple command asynchronously."""
    return f"Handled two: {cmd.data}"


@system(on_command=AnotherCommand)
def another_handler(cmd: AnotherCommand) -> int:
    """Handle the AnotherCommand."""
    return cmd.value * 2


@system(on_command=FailingCommand)
def failing_handler(_cmd: FailingCommand) -> None:
    """Handle a command that is designed to fail."""
    raise ValueError("This handler is designed to fail.")


@system(on_command=ListCommand)
def list_handler_one(_cmd: ListCommand) -> list[str]:
    """Handle a command by returning a list."""
    return ["a", "b"]


@system(on_command=ListCommand)
def list_handler_two(_cmd: ListCommand) -> list[str]:
    """Handle a command by returning another list."""
    return ["c", "d"]


@system(on_command=MismatchCommand)
def mismatch_handler(_cmd: MismatchCommand) -> str:  # Intentionally returns str, but command expects int
    """Handle a command with a mismatched return type."""
    return "this is not an int"


@system(on_command=StreamingCommand)
async def streaming_handler(_cmd: StreamingCommand) -> AsyncGenerator[CustomEvent, None]:
    """Handle a command by streaming custom events."""
    yield CustomEvent(message="First")
    yield CustomEvent(message="Second")


@system(on_command=LegacyCommand)
def legacy_handler(_cmd: Annotated[LegacyCommand, "Command"]) -> str:
    """Handle a command for backward compatibility testing."""
    return "Handled legacy command"


# Tests
@pytest.mark.asyncio
async def test_get_one_value_success(test_world_alpha: World) -> None:
    """Test CommandStream.get_one_value() for a single handler."""
    world = test_world_alpha
    world.register_system(system_func=another_handler)
    command = AnotherCommand(value=10)
    result = await world.dispatch_command(command).get_one_value()
    assert result == 20


@pytest.mark.asyncio
async def test_get_all_results_flat(test_world_alpha: World) -> None:
    """Test the get_all_results_flat method."""
    world = test_world_alpha
    world.register_system(system_func=list_handler_one)
    world.register_system(system_func=list_handler_two)
    command = ListCommand()
    results = await world.dispatch_command(command).get_all_results_flat()
    assert len(results) == 4
    assert set(results) == {"a", "b", "c", "d"}


@pytest.mark.asyncio
async def test_get_one_value_failure(test_world_alpha: World) -> None:
    """Test that CommandStream.get_one_value() fails for multiple handlers."""
    world = test_world_alpha
    world.register_system(system_func=simple_handler_one)
    world.register_system(system_func=simple_handler_two)
    command = SimpleCommand(data="test")
    with pytest.raises(ValueError, match=r"Expected one result, but found 2."):
        await world.dispatch_command(command).get_one_value()


@pytest.mark.asyncio
async def test_get_all_results(test_world_alpha: World) -> None:
    """Test CommandStream.get_all_results() for multiple handlers."""
    world = test_world_alpha
    world.register_system(system_func=simple_handler_one)
    world.register_system(system_func=simple_handler_two)
    command = SimpleCommand(data="test")
    results = await world.dispatch_command(command).get_all_results()
    assert len(results) == 2
    assert "Handled one: test" in results
    assert "Handled two: test" in results


@pytest.mark.asyncio
async def test_dispatch_command_with_no_handlers(test_world_alpha: World) -> None:
    """Test that dispatching a command with no handlers returns an empty result."""
    world = test_world_alpha
    command = SimpleCommand(data="unhandled")
    results = await world.dispatch_command(command).get_all_results()
    assert len(results) == 0


@pytest.mark.asyncio
async def test_command_handler_failure_propagates(test_world_alpha: World) -> None:
    """Test that if a handler raises an exception, it propagates."""
    world = test_world_alpha
    world.register_system(system_func=failing_handler)
    command = FailingCommand()
    with pytest.raises(ValueError, match=r"This handler is designed to fail."):
        await world.dispatch_command(command).get_one_value()


@pytest.mark.asyncio
async def test_dispatch_different_commands(test_world_alpha: World) -> None:
    """Test that the dispatcher correctly routes commands to their handlers."""
    world = test_world_alpha
    world.register_system(system_func=simple_handler_one)
    world.register_system(system_func=another_handler)

    res1 = await world.dispatch_command(SimpleCommand(data="dispatch one")).get_one_value()
    assert res1 == "Handled one: dispatch one"

    res2 = await world.dispatch_command(AnotherCommand(value=5)).get_one_value()
    assert res2 == 10


@pytest.mark.asyncio
async def test_list_return_values(test_world_alpha: World) -> None:
    """Test that handlers returning lists are correctly handled."""
    world = test_world_alpha
    world.register_system(system_func=list_handler_one)
    world.register_system(system_func=list_handler_two)
    command = ListCommand()
    results = await world.dispatch_command(command).get_all_results()
    assert len(results) == 2
    assert ["a", "b"] in results
    assert ["c", "d"] in results


@pytest.mark.asyncio
async def test_unified_streaming_handler(test_world_alpha: World) -> None:
    """Test that a handler that is an AsyncGenerator works with the unified dispatch_command function."""
    world = test_world_alpha
    world.register_system(system_func=streaming_handler)

    command = StreamingCommand()
    results = [item async for item in world.dispatch_command(command)]

    assert len(results) == 2
    assert isinstance(results[0], CustomEvent)
    assert results[0].message == "First"
    assert isinstance(results[1], CustomEvent)
    assert results[1].message == "Second"


@pytest.mark.asyncio
async def test_return_type_mismatch_raises_error(test_world_alpha: World) -> None:
    """Test that a TypeError is raised when a handler's return type annotation does not match the command's expected result type."""
    world = test_world_alpha
    with pytest.raises(TypeError, match="Return type mismatch for command 'MismatchCommand'"):
        world.register_system(system_func=mismatch_handler)


@pytest.mark.asyncio
async def test_string_identity_backward_compatibility(test_world_alpha: World) -> None:
    """Test that string-based identities for command injection still work."""
    world = test_world_alpha
    world.register_system(system_func=legacy_handler)
    command = LegacyCommand()
    result = await world.dispatch_command(command).get_one_value()
    assert result == "Handled legacy command"
