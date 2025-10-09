"""Saved files information class."""

from dataclasses import dataclass


@dataclass
class SavedFilesInfo:
    """Information about files saved during scraping operation."""

    main_content_file: str
    links_file: str
    metadata_file: str
    child_pages_files: list[str] | None
    total_files_saved: int
