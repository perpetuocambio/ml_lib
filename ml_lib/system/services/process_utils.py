"""
System process utilities for process management.

Provides cross-platform utilities for managing system processes.
"""

import logging
import psutil
from typing import Optional

logger = logging.getLogger(__name__)


class ProcessManager:
    """Utilities for managing system processes."""

    @staticmethod
    def kill_process_by_name(process_name: str, force: bool = False) -> bool:
        """
        Kill all processes matching a name.

        Args:
            process_name: Name of process to kill
            force: Use SIGKILL (force=True) or SIGTERM (force=False)

        Returns:
            True if at least one process was killed

        Example:
            >>> ProcessManager.kill_process_by_name("ollama", force=True)
        """
        killed_count = 0

        try:
            for proc in psutil.process_iter(["name", "pid"]):
                try:
                    if process_name.lower() in proc.info["name"].lower():
                        logger.info(
                            f"Killing process: {proc.info['name']} (PID: {proc.info['pid']})"
                        )

                        if force:
                            proc.kill()  # SIGKILL
                        else:
                            proc.terminate()  # SIGTERM

                        killed_count += 1

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            if killed_count > 0:
                logger.info(
                    f"Killed {killed_count} process(es) matching '{process_name}'"
                )
                return True
            else:
                logger.debug(f"No processes found matching '{process_name}'")
                return False

        except Exception as e:
            logger.error(f"Failed to kill processes: {e}")
            return False

    @staticmethod
    def is_process_running(process_name: str) -> bool:
        """
        Check if a process is running.

        Args:
            process_name: Name of process to check

        Returns:
            True if at least one matching process is running

        Example:
            >>> if ProcessManager.is_process_running("ollama"):
            ...     print("Ollama is running")
        """
        try:
            for proc in psutil.process_iter(["name"]):
                if process_name.lower() in proc.info["name"].lower():
                    return True
            return False
        except Exception as e:
            logger.error(f"Failed to check process: {e}")
            return False

    @staticmethod
    def get_process_memory(process_name: str) -> Optional[int]:
        """
        Get total memory usage of all processes with given name.

        Args:
            process_name: Name of process

        Returns:
            Total memory in bytes, or None if not found

        Example:
            >>> mem_bytes = ProcessManager.get_process_memory("ollama")
            >>> if mem_bytes:
            ...     print(f"Ollama using {mem_bytes / (1024**3):.2f} GB")
        """
        total_memory = 0
        found = False

        try:
            for proc in psutil.process_iter(["name", "memory_info"]):
                if process_name.lower() in proc.info["name"].lower():
                    total_memory += proc.info["memory_info"].rss
                    found = True

            return total_memory if found else None

        except Exception as e:
            logger.error(f"Failed to get process memory: {e}")
            return None

    @staticmethod
    def get_process_pids(process_name: str) -> list[int]:
        """
        Get all PIDs matching a process name.

        Args:
            process_name: Name of process

        Returns:
            List of PIDs

        Example:
            >>> pids = ProcessManager.get_process_pids("python")
            >>> print(f"Found {len(pids)} Python processes")
        """
        pids = []

        try:
            for proc in psutil.process_iter(["name", "pid"]):
                if process_name.lower() in proc.info["name"].lower():
                    pids.append(proc.info["pid"])

            return pids

        except Exception as e:
            logger.error(f"Failed to get process PIDs: {e}")
            return []
