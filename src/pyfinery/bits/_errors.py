# SPDX-FileCopyrightText: 2026 zacharyvmm <58368863+zacharyvmm@users.noreply.github.com>
#
# SPDX-License-Identifier: EPL-2.0

"""Errors of the bits engine."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._syntax import Position


class RuleError(ValueError):
    """Rules the bits engine can't evaluate: malformed, or naming what isn't there."""

    def __init__(self, message: str, position: Position | None = None) -> None:
        self.message = message
        self.position = position
        super().__init__(f"{position}: {message}" if position is not None else message)


class UnsupportedRuleError(RuleError):
    """Rules using a part of Refinery's language the bits engine doesn't evaluate."""

    def __init__(self, feature: str, position: Position | None = None) -> None:
        self.feature = feature
        super().__init__(f"the bits engine doesn't support {feature}", position)
