"""Errors that distinguish missing memory from damaged or foreign files."""


class MemoryNotFoundError(LookupError):
    """The requested graph, episode, or node does not exist."""


class MemoryIntegrityError(RuntimeError):
    """The file is not this graph's supported memory database."""
