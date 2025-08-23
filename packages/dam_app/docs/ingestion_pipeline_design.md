# Event-Driven Ingestion Pipeline Design

This document outlines the design for a new event-driven ingestion pipeline for the `dam` system. This approach is chosen for its ability to handle transient data (like in-memory file streams) gracefully and for its high degree of modularity and extensibility.

## 1. Core Concept: A Chain of Events

The pipeline is orchestrated through a chain of events. A process is initiated by a single, generic event. Systems listen for this event, perform a specific task, and then may fire new, more specific events. This creates a branching, composable workflow that is easy to extend.

The primary benefits of this approach are:
- **Decoupling**: Systems don't call each other directly. They only need to know about the events they consume and produce.
- **State Passing**: Transient data, such as the `file_content` as `bytes`, is passed along within the event payloads, avoiding the need to write to disk and re-read between steps.
- **Extensibility**: To add a new processing step or support a new file type, a developer simply creates a new event and a new system to handle it.

## 2. Event Definitions and Locations

To maintain modularity, events will be defined in the packages most relevant to them.

### Core `dam` Package Events
- **File Location**: `packages/dam/src/dam/core/events.py`
- **Event**: `AssetIngestionRequested(BaseEvent)`
  - **Payload**:
    - `entity: Entity`: The newly created entity for this asset.
    - `file_content: bytes`: The raw file content.
    - `original_filename: str`: The original name of the file.
  - **Purpose**: This is the entry point for the entire pipeline. It signals that a new asset needs to be processed from an in-memory stream.

### `dam_fs` Package Events
- **File Location**: `packages/dam_fs/src/dam_fs/events.py`
- **Event**: `FileStored(BaseEvent)`
  - **Payload**:
    - `entity: Entity`: The entity the file is associated with.
    - `file_id: int`: The ID of the `File` record in the database.
    - `file_path: Path`: The path to the file in the content-addressable storage.
  - **Purpose**: Fired after the initial `AssetIngestionRequested` event has been handled and the file has been successfully saved to the CAS (Content-Addressable Storage). This signals the transition from in-memory processing to processing based on a persisted file.

### `dam_media_image` Plugin Events
- **File Location**: `packages/dam_media_image/src/dam_media_image/events.py`
- **Event**: `ImageAssetDetected(BaseEvent)`
  - **Payload**:
    - `entity: Entity`: The asset's entity.
    - `file_id: int`: The ID of the associated `File` record.
  - **Purpose**: Fired by a dispatcher system when it determines that a stored file is an image. This triggers image-specific processing systems.

### `dam_psp` Plugin Events
- **File Location**: `packages/dam_psp/src/dam_psp/events.py`
- **Event**: `PspIsoAssetDetected(BaseEvent)`
  - **Payload**:
    - `entity: Entity`: The asset's entity.
    - `file_id: int`: The ID of the associated `File` record.
  - **Purpose**: Fired when a stored file is identified as a PSP ISO, triggering PSP-specific systems.

## 3. System Definitions and Locations

Systems are the workers in the pipeline, listening for events and performing actions.

### Core `dam` / `dam_app` Systems
- **System**: `ingestion_request_system`
  - **Listens for**: `AssetIngestionRequested`
  - **Actions**:
    1. Calculates content hashes of `file_content`.
    2. Stores the file in the Content-Addressable Storage via `FileStorageResource`.
    3. Adds a `File` component to the entity.
    4. Fires the `FileStored` event with the new `file_id` and `file_path`.
  - **Location**: `packages/dam_app/src/dam_app/systems/ingestion_systems.py` (in `dam_app` as it orchestrates the initial ingestion).

- **System**: `asset_dispatcher_system`
  - **Listens for**: `FileStored`
  - **Actions**:
    1. Uses the `file_path` to determine the MIME type of the file.
    2. Based on the MIME type, fires the appropriate specific event (e.g., `ImageAssetDetected`, `PspIsoAssetDetected`).
  - **Location**: `packages/dam_app/src/dam_app/systems/ingestion_systems.py`.

### Plugin Systems

- **System**: `process_image_metadata_system` (`dam_media_image`)
  - **Listens for**: `ImageAssetDetected`
  - **Actions**:
    1. **Skip Logic**: Checks if an `ImageMetadataComponent` already exists for the entity. If so, it stops.
    2. Retrieves the file path using the `file_id`.
    3. Extracts image metadata (dimensions, format, etc.).
    4. Adds an `ImageMetadataComponent` to the entity.

- **System**: `process_psp_iso_system` (`dam_psp`)
  - **Listens for**: `PspIsoAssetDetected`
  - **Actions**:
    1. **Skip Logic**: Checks if a `PspIsoInfoComponent` already exists.
    2. Retrieves the file path.
    3. Extracts information from the ISO.
    4. Adds a `PspIsoInfoComponent` to the entity.

## 4. Example Workflow: Ingesting a New Image

1. **Entry Point**: An external caller (e.g., a CLI command, a web endpoint) has an image as `bytes`. It creates an `Entity` and fires an `AssetIngestionRequested` event with the entity and file content.
2. **File Storage**: The `ingestion_request_system` listens for this event. It saves the file to the CAS, adds a `File` component, and fires a `FileStored` event.
3. **Dispatch**: The `asset_dispatcher_system` listens for `FileStored`. It checks the file's MIME type and determines it's an image. It then fires an `ImageAssetDetected` event.
4. **Image Processing**: The `process_image_metadata_system` in the `dam_media_image` plugin listens for `ImageAssetDetected`. It checks that no `ImageMetadataComponent` exists, then proceeds to extract metadata and add the component to the entity.
5. **Completion**: The pipeline branch for this image is now complete. The session is committed, and the entity is now fully described in the database.

This design provides a robust, modular, and understandable foundation for the ingestion process.
