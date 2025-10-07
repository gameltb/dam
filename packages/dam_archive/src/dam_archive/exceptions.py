"""Custom exceptions for archive handling."""


class ArchiveError(Exception):
    """Base class for archive-related errors."""

    pass


class UnsupportedArchiveError(ArchiveError):
    """Raised when the archive format is not supported by the handler."""

    pass


class PasswordRequiredError(ArchiveError):
    """Raised when an archive requires a password but none was provided."""

    pass


class InvalidPasswordError(PasswordRequiredError):
    """Raised when an incorrect password is used for an archive."""

    pass
