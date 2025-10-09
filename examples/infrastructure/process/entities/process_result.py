"""Process execution result."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProcessResult:
    """Result of executing a process command."""

    stdout: str
    stderr: str
    returncode: int
