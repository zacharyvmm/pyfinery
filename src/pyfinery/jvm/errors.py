"""Errors of :mod:`pyfinery.jvm`."""

from __future__ import annotations

from ..errors import RefineryError


class JvmError(RefineryError):
    """Refinery's JVM could not be found, set up or started, or was asked for what it was not
    started with."""
