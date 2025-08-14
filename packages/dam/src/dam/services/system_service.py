import logging

logger = logging.getLogger(__name__)

# This file previously contained system-related functions that are now being
# refactored into a more structured systems approach using dam.core.systems,
# dam.core.stages, dam.core.resources, and dam.core.system_params.

# Individual systems (like metadata extraction) are being moved to their own
# modules under the `dam.systems` package (e.g., `dam.systems.metadata_systems`).

# The WorldScheduler in `dam.core.systems` will be responsible for discovering,
# injecting dependencies into, and executing these systems based on stages
# and component markers or events.

# Old functions like `process_entity_with_systems`,
# `metadata_extraction_system_process_entity`, and helper utilities
# have been removed or relocated.

# This file can be removed if no longer needed, or repurposed for
# higher-level system orchestration logic if a different pattern emerges.
# For now, it serves as a placeholder indicating the refactoring.

logger.info("dam.services.system_service is being refactored. Systems are moving to dam.core.systems and dam.systems.*")
