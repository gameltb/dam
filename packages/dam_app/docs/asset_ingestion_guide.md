# Application Guide: Asset Ingestion Workflow

This guide demonstrates how to build a robust asset ingestion workflow using the DAM ECS framework, focusing on the new command system for orchestration and returning results.

## Overview

The goal of our asset ingestion workflow is to import a file, process it based on its type, and avoid duplicating work if the asset has already been imported. We will use a series of commands and handlers to achieve this in a decoupled and extensible way.

The high-level logic is as follows:
1.  Initiate an `IngestFileCommand` with a file path.
2.  A primary handler checks if the file path has been processed before. If so, it returns an "Already Imported" status.
3.  If not, it calculates the file's hash and dispatches a new `ProcessFileByHashCommand`.
4.  A handler for `ProcessFileByHashCommand` checks if the hash already exists in the database. If so, it returns a "Duplicate Hash" status.
5.  If the hash is new, it determines the file type and dispatches a type-specific command (e.g., `ProcessImageCommand`, `ProcessVideoCommand`).
6.  Specific handlers for these commands perform the actual processing (e.g., generating thumbnails, extracting metadata).
7.  The results from each step are returned to the original caller.

## Core Concepts: Events vs. Commands

Before we dive in, it's important to understand the two main ways systems communicate:

-   **Events (`BaseEvent`)**: These are "fire-and-forget" notifications. They are useful for broadcasting that something has happened without needing a response. For example, sending a `FileImportedEvent` after an import is complete.
-   **Commands (`BaseCommand`)**: These are requests that expect a response. They are used when you need to orchestrate a workflow and get a result back from one or more handlers. Our ingestion pipeline will be built using commands.

## Step 1: Defining the Commands

First, we define the commands we need for our workflow.

```python
# in packages/dam_app/src/dam_app/commands.py
from dataclasses import dataclass
from pathlib import Path
from dam.commands.core import BaseCommand

@dataclass
class IngestFileCommand(BaseCommand):
    """Initiates the ingestion process for a file at a given path."""
    file_path: Path

@dataclass
class ProcessFileByHashCommand(BaseCommand):
    """Processes a file based on its content hash."""
    file_path: Path
    file_hash: str

@dataclass
class ProcessImageCommand(BaseCommand):
    """Handles the specific processing for an image file."""
    file_path: Path
    file_hash: str
```

## Step 2: Implementing the Handlers

Next, we create the handlers for these commands. These are just regular systems decorated with `@handles_command`.

### Ingestion Initiator

This handler is the entry point. It checks for existing imports by path and then delegates to the hash processing command.

```python
# in packages/dam_app/src/dam_app/systems/ingestion.py
from dam.core.systems import handles_command
from dam.core.world import World
from dam_app.commands import IngestFileCommand, ProcessFileByHashCommand
from dam_app.functions import file_functions, hash_functions

@handles_command(IngestFileCommand)
async def handle_ingest_file(cmd: IngestFileCommand, world: World):
    """
    Handles the initial file ingestion request.
    """
    # 1. Check if this file path has been imported before
    if await file_functions.is_path_imported(world, cmd.file_path):
        return {"status": "SKIPPED", "reason": "Path already imported"}

    # 2. Calculate the file hash
    file_hash = await hash_functions.calculate_hash(cmd.file_path)

    # 3. Dispatch a new command to process by hash
    process_cmd = ProcessFileByHashCommand(
        file_path=cmd.file_path,
        file_hash=file_hash
    )
    # We dispatch a new command and return its result to the original caller
    return await world.dispatch_command(process_cmd)
```

### Hash and Type-Based Processing

This handler checks for duplicate content via hash and then routes to a type-specific handler.

```python
# in packages/dam_app/src/dam_app/systems/ingestion.py
@handles_command(ProcessFileByHashCommand)
async def handle_process_by_hash(cmd: ProcessFileByHashCommand, world: World):
    """
    Handles processing a file based on its hash, checking for duplicates.
    """
    # 1. Check if this hash exists in the database
    if await hash_functions.does_hash_exist(world, cmd.file_hash):
        return {"status": "SKIPPED", "reason": "Duplicate hash"}

    # 2. Determine file type and dispatch to a specific processor
    file_type = file_functions.get_file_type(cmd.file_path)

    if file_type == "image":
        image_cmd = ProcessImageCommand(file_path=cmd.file_path, file_hash=cmd.file_hash)
        return await world.dispatch_command(image_cmd)
    elif file_type == "video":
        # Dispatch a ProcessVideoCommand, etc.
        pass

    return {"status": "FAILED", "reason": "Unsupported file type"}
```

### Image-Specific Processor

This is where the actual work for a specific file type happens.

```python
# in packages/dam_media_image/src/dam_media_image/systems.py
from dam.core.systems import handles_command
from dam_app.commands import ProcessImageCommand # Assuming commands are accessible
from .functions import thumbnail_functions

@handles_command(ProcessImageCommand)
async def handle_process_image(cmd: ProcessImageCommand):
    """
    Performs image-specific processing, like generating thumbnails.
    """
    thumbnail_path = await thumbnail_functions.generate(cmd.file_path)

    # Here you would also create the asset, components, etc. in the database

    return {"status": "SUCCESS", "thumbnail_path": str(thumbnail_path)}
```

## Step 3: Registering Handlers and Running the Workflow

Finally, you need to register your handlers with the `World` and then you can dispatch commands.

```python
# In your application setup
from dam_app.systems.ingestion import handle_ingest_file, handle_process_by_hash
from dam_media_image.systems import handle_process_image

def setup_world(world: World):
    # Register all your command handlers
    world.register_system(handle_ingest_file, command_type=IngestFileCommand)
    world.register_system(handle_process_by_hash, command_type=ProcessFileByHashCommand)
    world.register_system(handle_process_image, command_type=ProcessImageCommand)

# To run the workflow
async def run_ingestion(world: World, file_to_ingest: Path):
    command = IngestFileCommand(file_path=file_to_ingest)
    result = await world.dispatch_command(command)

    # The result will be a CommandResult object containing the final result
    # from the chain of command handlers.
    print(result.results)
```

This example shows how the command system allows for building complex, multi-step workflows in a clean, decoupled manner, while still allowing results to be passed back to the initial caller. Each handler has a single responsibility, and the workflow can be easily extended by adding new commands and handlers.
