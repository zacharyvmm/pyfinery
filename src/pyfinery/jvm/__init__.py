"""Refinery in this process: its Java library, driven through JPype.

Install the ``jvm`` extra (``pyfinery[jvm]``) and a JDK 25 or later. The
first use fetches Refinery's jars from Maven Central into a cache directory
and checks them against the hashes pyfinery pins; later uses start from the
cache.

A process has one JVM, and what it can do is fixed when it starts: by
default, it validates and queries. To generate models too, say so before
anything starts it::

    import pyfinery.jvm
    pyfinery.jvm.configure(generation=True)

:func:`start` starts it explicitly; the engines that need it start it
themselves on first use.

To validate and query on Refinery's own matcher, pass a :class:`JvmEngine`
where :meth:`~pyfinery.Metamodel.validate` takes a client; to validate a graph
as it changes, keep it in a :class:`Session`.
"""

from __future__ import annotations

from ._matching.engine import JvmEngine
from ._matching.session import Changes, Checkpoint, Session
from ._runtime import (
    MINIMUM_VERSION,
    Settings,
    Status,
    configure,
    helpers,
    require,
    settings,
    start,
    started,
    status,
)
from .errors import JvmError, UnsupportedRuleError

__all__ = [
    "Changes",
    "Checkpoint",
    "JvmEngine",
    "JvmError",
    "MINIMUM_VERSION",
    "Session",
    "Settings",
    "Status",
    "UnsupportedRuleError",
    "configure",
    "helpers",
    "require",
    "settings",
    "start",
    "started",
    "status",
]
