class PasswordRequiredError(Exception):
    """
    Raised when an archive requires a password but none was provided or the provided ones were incorrect.
    """

    pass
