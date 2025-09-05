from dataclasses import dataclass

import pytest
from _pytest.logging import LogCaptureFixture

from dam.core.commands import BaseCommand
from dam.core.systems import system
from dam.core.world import World


# Define some test commands
@dataclass
class SimpleCommand(BaseCommand[str]):
    data: str


@dataclass
class AnotherCommand(BaseCommand[int]):
    value: int


@dataclass
class FailingCommand(BaseCommand[None]):
    pass


@dataclass
class ListCommand(BaseCommand[list[str]]):
    pass


@dataclass
class MismatchCommand(BaseCommand[int]):
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


# Tests
@pytest.mark.asyncio
async def test_register_and_dispatch_single_handler(test_world_alpha: World) -> None:
    """
    Tests that a single command handler can be registered and successfully
    processes a command, returning a result.
    """
    world = test_world_alpha
    world.register_system(system_func=another_handler)

    command = AnotherCommand(value=10)
    result = await world.dispatch_command(command)

    assert result is not None
    assert len(result.results) == 1
    assert result.results[0].is_ok()
    assert result.results[0].unwrap() == 20
    assert result.get_one_value() == 20


@pytest.mark.asyncio
async def test_dispatch_multiple_handlers(test_world_alpha: World) -> None:
    """
    Tests that multiple handlers for the same command all execute and their
    results are collected.
    """
    world = test_world_alpha
    world.register_system(system_func=simple_handler_one)
    world.register_system(system_func=simple_handler_two)

    command = SimpleCommand(data="test")
    result = await world.dispatch_command(command)

    assert result is not None
    assert len(result.results) == 2

    ok_values = list(result.iter_ok_values())
    assert len(ok_values) == 2
    assert "Handled one: test" in ok_values
    assert "Handled two: test" in ok_values


@pytest.mark.asyncio
async def test_dispatch_command_with_no_handlers(test_world_alpha: World) -> None:
    """
    Tests that dispatching a command with no registered handlers returns an
    empty result and does not raise an error.
    """
    world = test_world_alpha
    # Note: No handlers are registered for SimpleCommand in this test's world instance

    command = SimpleCommand(data="unhandled")
    result = await world.dispatch_command(command)

    assert result is not None
    assert len(result.results) == 0


@pytest.mark.asyncio
async def test_command_handler_failure_is_captured(test_world_alpha: World) -> None:
    """
    Tests that if a command handler raises an exception, the exception is
    captured in a HandlerResult and does not abort the dispatch.
    """
    world = test_world_alpha
    world.register_system(system_func=failing_handler)
    world.register_system(system_func=simple_handler_one)  # Register another handler to ensure it runs

    command = FailingCommand()
    result = await world.dispatch_command(command)

    # The dispatch itself should not raise an error
    assert result is not None
    assert len(result.results) == 1

    handler_res = result.results[0]
    assert handler_res.is_err()
    assert isinstance(handler_res.exception, ValueError)

    with pytest.raises(ValueError, match="This handler is designed to fail."):
        handler_res.unwrap()

    # Test the convenience methods
    with pytest.raises(ValueError, match="No successful results found."):
        result.get_first_ok_value()

    with pytest.raises(ValueError, match="Expected one result, but found none."):
        result.get_one_value()


@pytest.mark.asyncio
async def test_dispatch_different_commands(test_world_alpha: World) -> None:
    """
    Tests that the command dispatcher correctly routes commands to their
    specific handlers and not to handlers for other commands.
    """
    world = test_world_alpha
    world.register_system(system_func=simple_handler_one)
    world.register_system(system_func=another_handler)

    # Dispatch first command
    cmd1 = SimpleCommand(data="dispatch one")
    res1 = await world.dispatch_command(cmd1)
    assert res1.get_one_value() == "Handled one: dispatch one"

    # Dispatch second command
    cmd2 = AnotherCommand(value=5)
    res2 = await world.dispatch_command(cmd2)
    assert res2.get_one_value() == 10


@pytest.mark.asyncio
async def test_iter_ok_values_flat(test_world_alpha: World) -> None:
    """
    Tests the iter_ok_values_flat method.
    """
    world = test_world_alpha
    world.register_system(system_func=list_handler_one)
    world.register_system(system_func=list_handler_two)

    command = ListCommand()
    result = await world.dispatch_command(command)

    assert result is not None
    assert len(result.results) == 2

    flat_values = list(result.iter_ok_values_flat())
    assert len(flat_values) == 4
    assert set(flat_values) == {"a", "b", "c", "d"}


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
