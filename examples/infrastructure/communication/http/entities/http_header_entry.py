"""HTTP header entry - single name/value pair."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HttpHeaderEntry:
    """A single HTTP header name/value pair."""

    name: str
    value: str
