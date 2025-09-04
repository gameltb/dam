from dataclasses import dataclass

import pytest

from dam.core.commands import BaseCommand
from dam.core.exceptions import CommandHandlingError
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
def failing_handler(cmd: FailingCommand):
    raise ValueError("This handler is designed to fail.")


# Tests
@pytest.mark.asyncio
async def test_register_and_dispatch_single_handler(test_world_alpha: World):
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
    assert result.results[0] == 20


@pytest.mark.asyncio
async def test_dispatch_multiple_handlers(test_world_alpha: World):
    """
    Tests that the first handler to return a non-None result wins.
    """
    world = test_world_alpha
    world.register_system(system_func=simple_handler_one)
    world.register_system(system_func=simple_handler_two)

    command = SimpleCommand(data="test")
    result = await world.dispatch_command(command)

    assert result is not None
    assert len(result.results) == 1
    assert result.results[0] == "Handled one: test"


@pytest.mark.asyncio
async def test_dispatch_command_with_no_handlers(test_world_alpha: World):
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
async def test_command_handler_failure(test_world_alpha: World):
    """
    Tests that if a command handler raises an exception, the dispatch
    is aborted and a CommandHandlingError is raised.
    """
    world = test_world_alpha
    world.register_system(system_func=failing_handler)

    command = FailingCommand()

    with pytest.raises(CommandHandlingError) as exc_info:
        await world.dispatch_command(command)

    assert exc_info.value.command_type == "FailingCommand"
    # The new design does not propagate the handler name to the top-level exception
    # assert exc_info.value.handler_name == "failing_handler"
    assert isinstance(exc_info.value.original_exception, ValueError)


@pytest.mark.asyncio
async def test_dispatch_different_commands(test_world_alpha: World):
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
    assert len(res1.results) == 1
    assert res1.results[0] == "Handled one: dispatch one"

    # Dispatch second command
    cmd2 = AnotherCommand(value=5)
    res2 = await world.dispatch_command(cmd2)
    assert len(res2.results) == 1
    assert res2.results[0] == 10
