from dataclasses import dataclass
from typing import AsyncGenerator

import pytest
from _pytest.logging import LogCaptureFixture

from dam.core.commands import BaseCommand
from dam.core.systems import system
from dam.core.world import World
from dam.system_events import BaseSystemEvent


# Define some test commands
@dataclass
class SimpleCommand(BaseCommand[str, BaseSystemEvent]):
    data: str


@dataclass
class AnotherCommand(BaseCommand[int, BaseSystemEvent]):
    value: int


@dataclass
class FailingCommand(BaseCommand[None, BaseSystemEvent]):
    pass


@dataclass
class ListCommand(BaseCommand[list[str], BaseSystemEvent]):
    pass


@dataclass
class MismatchCommand(BaseCommand[int, BaseSystemEvent]):
    pass


@dataclass
class CustomEvent(BaseSystemEvent):
    message: str


@dataclass
class StreamingCommand(BaseCommand[None, CustomEvent]):
    pass


# Define some test systems (command handlers)
@system(on_command=SimpleCommand)
def simple_handler_one(cmd: SimpleCommand) -> str:
    return f"Handled one: {cmd.data}"


@system(on_command=SimpleCommand)
async def simple_handler_two(cmd: SimpleCommand) -> str:
    return f"Handled two: {cmd.data}"


@system(on_command=AnotherCommand)
def another_handler(cmd: AnotherCommand) -> int:
    return cmd.value * 2


@system(on_command=FailingCommand)
def failing_handler(cmd: FailingCommand) -> None:
    raise ValueError("This handler is designed to fail.")


@system(on_command=ListCommand)
def list_handler_one(cmd: ListCommand) -> list[str]:
    return ["a", "b"]


@system(on_command=ListCommand)
def list_handler_two(cmd: ListCommand) -> list[str]:
    return ["c", "d"]


@system(on_command=MismatchCommand)
def mismatch_handler(cmd: MismatchCommand) -> str:  # Intentionally returns str, but command expects int
    return "this is not an int"


@system(on_command=StreamingCommand)
async def streaming_handler(cmd: StreamingCommand) -> AsyncGenerator[CustomEvent, None]:
    yield CustomEvent(message="First")
    yield CustomEvent(message="Second")


# Tests
@pytest.mark.asyncio
async def test_get_one_value_success(test_world_alpha: World) -> None:
    """Tests CommandStream.get_one_value() for a single handler."""
    world = test_world_alpha
    world.register_system(system_func=another_handler)
    command = AnotherCommand(value=10)
    result = await world.dispatch_command(command).get_one_value()
    assert result == 20


@pytest.mark.asyncio
async def test_get_all_results_flat(test_world_alpha: World) -> None:
    """Tests the get_all_results_flat method."""
    world = test_world_alpha
    world.register_system(system_func=list_handler_one)
    world.register_system(system_func=list_handler_two)
    command = ListCommand()
    results = await world.dispatch_command(command).get_all_results_flat()
    assert len(results) == 4
    assert set(results) == {"a", "b", "c", "d"}


@pytest.mark.asyncio
async def test_get_one_value_failure(test_world_alpha: World) -> None:
    """Tests that CommandStream.get_one_value() fails for multiple handlers."""
    world = test_world_alpha
    world.register_system(system_func=simple_handler_one)
    world.register_system(system_func=simple_handler_two)
    command = SimpleCommand(data="test")
    with pytest.raises(ValueError, match="Expected one result, but found 2."):
        await world.dispatch_command(command).get_one_value()


@pytest.mark.asyncio
async def test_get_all_results(test_world_alpha: World) -> None:
    """Tests CommandStream.get_all_results() for multiple handlers."""
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
    """Tests that dispatching a command with no handlers returns an empty result."""
    world = test_world_alpha
    command = SimpleCommand(data="unhandled")
    results = await world.dispatch_command(command).get_all_results()
    assert len(results) == 0


@pytest.mark.asyncio
async def test_command_handler_failure_propagates(test_world_alpha: World) -> None:
    """Tests that if a handler raises an exception, it propagates."""
    world = test_world_alpha
    world.register_system(system_func=failing_handler)
    command = FailingCommand()
    with pytest.raises(ValueError, match="This handler is designed to fail."):
        await world.dispatch_command(command).get_one_value()


@pytest.mark.asyncio
async def test_dispatch_different_commands(test_world_alpha: World) -> None:
    """Tests that the dispatcher correctly routes commands to their handlers."""
    world = test_world_alpha
    world.register_system(system_func=simple_handler_one)
    world.register_system(system_func=another_handler)

    res1 = await world.dispatch_command(SimpleCommand(data="dispatch one")).get_one_value()
    assert res1 == "Handled one: dispatch one"

    res2 = await world.dispatch_command(AnotherCommand(value=5)).get_one_value()
    assert res2 == 10


@pytest.mark.asyncio
async def test_list_return_values(test_world_alpha: World) -> None:
    """Tests that handlers returning lists are correctly handled."""
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
    """
    Tests that a handler that is an AsyncGenerator works with the unified
    dispatch_command function.
    """
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
async def test_return_type_mismatch_warning(test_world_alpha: World, caplog: LogCaptureFixture) -> None:
    """
    Tests that a warning is logged when a handler's return type annotation
    does not match the command's expected result type.
    """
    world = test_world_alpha
    with caplog.at_level("WARNING"):
        world.register_system(system_func=mismatch_handler)

    assert len(caplog.records) == 1
    assert "Potential return type mismatch" in caplog.text
    assert "command 'MismatchCommand'" in caplog.text
    assert "Handler 'mismatch_handler'" in caplog.text
    assert "expects '<class 'int'>'" in caplog.text
    assert "annotated to return '<class 'str'>'" in caplog.text
