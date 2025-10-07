"""Defines constants for the `source_type` field in the OriginalSourceInfoComponent."""

# Indicates the source was a local file that was processed and ingested.
# The file's content is typically now managed by the DAM.
SOURCE_TYPE_LOCAL_FILE = "local_file"

# Indicates the source is a file that is referenced by its path, but its content
# might not be directly managed or stored by the DAM. For example, a file on a
# network share that is cataloged but not copied.
SOURCE_TYPE_REFERENCED_FILE = "referenced_file"

# Indicates the source was a web URL. The content might have been downloaded,
# or the URL itself is the reference.
SOURCE_TYPE_WEB_SOURCE = "web_source"

# Indicates that the primary file associated with the entity (whose properties
# are typically stored in a FilePropertiesComponent on the same entity)
# is itself the original source. This is common for direct uploads or when
# an asset is created directly from a file.
SOURCE_TYPE_PRIMARY_FILE = "primary_file"
