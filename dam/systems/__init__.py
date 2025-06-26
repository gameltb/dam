# This file makes the 'dam.systems' package.
# Import all system modules here to ensure their @system decorators run and register the systems.

# from . import metadata_systems # No longer needed here if cli.py imports specific system modules

# from . import hashing_systems # Example for future
# from . import query_systems   # Example for future
# from . import analysis_systems # Example for future

# Optionally, provide a way to list or access registered systems if needed directly,
# though usually the WorldScheduler would handle that internally.
# For now, simply importing is enough for registration via decorators.
print("dam.systems package initialized, systems (like metadata_systems) should be registered.")
