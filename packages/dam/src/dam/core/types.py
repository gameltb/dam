from typing import BinaryIO, Callable

# A function that returns a new, readable binary stream.
# Each call should produce a fresh stream, positioned at the beginning.
StreamProvider = Callable[[], BinaryIO]
