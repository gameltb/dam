# Domarkx Design Specification

This document outlines the design and architecture of the `domarkx` system.

## Core Concepts

`Domarkx` is designed to make Markdown documentation executable. It achieves this by parsing Markdown files, identifying code blocks, and executing them in a controlled environment. The key components of the system are:

-   **Document Parser**: Responsible for reading and parsing Markdown files.
-   **Code Block Executor**: Executes code blocks using various execution engines (e.g., local shell, remote Jupyter kernel).
-   **Agent**: An LLM-powered agent that can interact with the user and execute commands.
-   **CLI**: A command-line interface for interacting with the `domarkx` system.

## Architecture

The system is designed to be modular and extensible. The core components are loosely coupled and can be replaced or extended as needed.

-   **Actions**: The `action` module contains the core logic for executing documents and code blocks.
-   **Models**: The `models` module contains data models for representing documents, code blocks, and other entities.
-   **Tools**: The `tools` module contains a collection of tools that can be used by the agent.

## File Structure

A `domarkx` project has the following directory structure:

-   **sessions/**: Contains all the session files. Each session is a Markdown file.
-   **templates/**: Contains session templates.
-   **ProjectManager.md**: The main session for the Project Manager.

## Macro Commands

`domarkx` supports macro commands, which are special links in Markdown that can be used to perform actions. The format of a macro command is as follows:

`[@link_text](domarkx://command?param1=value1&param2=value2)`

-   `@link_text`: The display text of the macro.
-   `command`: The name of the command to execute.
-   `param1=value1&param2=value2`: The parameters for the command.

For example, to create a new session from a template, you can use the following macro:

`[@create_session](domarkx://create_session?template_name=my_template&session_name=new_session)`

## Future Development

-   **Support for more languages and execution engines.**
-   **Improved agent capabilities.**
-   **A more robust and user-friendly UI.**
