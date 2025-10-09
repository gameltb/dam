"""Digital Asset Manager (DAM) Core Library."""

from dam.core.world_manager import WorldManager

# Global singleton instance of the WorldManager.
# This instance should be used by applications to interact with DAM worlds.
world_manager = WorldManager()
