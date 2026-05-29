"""Exceptions for cp4dm_cpmpy_oscar_ml."""


class OscarMLError(Exception):
    """Base exception for this package."""


class NotSupportedError(OscarMLError):
    """Raised when an Oscar ML global is used without the native solver."""


class InconsistencyError(OscarMLError):
    """Raised when propagation detects failure (analogous to Oscar's Inconsistency)."""


class InvalidFormatError(OscarMLError):
    """Raised when a data file does not match the expected format."""


class UnsupportedExpressionError(OscarMLError):
    """Raised when CPM_oscar_ml encounters an expression it cannot handle."""
