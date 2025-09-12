# DAM Archive Plugin

This package is a plugin for the [Digital Asset Management (DAM)](../dam/README.md) system that provides functionalities for handling archive files.

## Features

- **Archive Format Support:** Provides handlers for various archive formats, allowing the DAM system to treat them as containers for other assets.
  - ZIP (`.zip`)
  - RAR (`.rar`)
- **Archive Member Ingestion:** Includes a system for extracting members from an archive, ingesting them as individual assets, and linking them back to the parent archive entity.
- **Password Protected Archives:** Supports handling password-protected archives.

## How it Works

The plugin registers `ArchiveHandler` implementations for the supported formats. When an archive file is ingested into the DAM, this plugin's systems are triggered to:
1.  Identify the archive type.
2.  Extract metadata about the files contained within the archive.
3.  Optionally, extract each member file, ingest it as a new asset, and create a `ArchiveMemberComponent` to link it to the original archive asset.
