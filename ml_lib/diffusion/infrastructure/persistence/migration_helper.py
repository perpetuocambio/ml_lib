"""Migration Helper - Convert from legacy ModelRegistry to SQLiteModelRepository.

Provides utilities to migrate data from old persistence layer to new clean architecture.
"""

from pathlib import Path
from typing import Optional
import logging

from ml_lib.diffusion.infrastructure.persistence.sqlite_model_repository import (
    SQLiteModelRepository,
)
from ml_lib.diffusion.infrastructure.persistence.model_registry_adapter import (
    ModelRegistryAdapter,
)

logger = logging.getLogger(__name__)


class RepositoryMigrationHelper:
    """
    Helper class for migrating from legacy ModelRegistry to SQLiteModelRepository.

    Usage:
        # Create instances
        legacy_registry = ModelRegistry(...)
        sqlite_repo = SQLiteModelRepository(db_path="loras.db")

        # Migrate
        helper = RepositoryMigrationHelper(
            source=legacy_registry,
            target=sqlite_repo
        )
        migrated_count = helper.migrate_all()
    """

    def __init__(
        self,
        source: ModelRegistryAdapter,
        target: SQLiteModelRepository,
    ):
        """
        Initialize migration helper.

        Args:
            source: Source repository (legacy ModelRegistry wrapped in adapter)
            target: Target SQLite repository
        """
        self.source = source
        self.target = target

    def migrate_all(self, skip_existing: bool = True) -> int:
        """
        Migrate all LoRAs from source to target.

        Args:
            skip_existing: If True, skip LoRAs that already exist in target

        Returns:
            Number of LoRAs migrated

        Raises:
            Exception: If migration fails
        """
        logger.info("Starting migration from legacy registry to SQLite...")

        try:
            all_loras = self.source.get_all_loras()
            logger.info(f"Found {len(all_loras)} LoRAs in source repository")

            migrated_count = 0
            skipped_count = 0
            error_count = 0

            for lora in all_loras:
                try:
                    # Check if already exists
                    if skip_existing:
                        existing = self.target.get_lora_by_name(lora.name)
                        if existing:
                            logger.debug(f"Skipping existing LoRA: {lora.name}")
                            skipped_count += 1
                            continue

                    # Add to target
                    self.target.add_lora(lora)
                    migrated_count += 1
                    logger.debug(f"Migrated LoRA: {lora.name}")

                except Exception as e:
                    logger.error(f"Error migrating LoRA {lora.name}: {e}")
                    error_count += 1

            logger.info(
                f"Migration complete: {migrated_count} migrated, "
                f"{skipped_count} skipped, {error_count} errors"
            )

            return migrated_count

        except Exception as e:
            logger.error(f"Migration failed: {e}")
            raise

    def verify_migration(self) -> tuple[int, int, list[str]]:
        """
        Verify that migration was successful.

        Returns:
            Tuple of (source_count, target_count, missing_names)
        """
        source_loras = self.source.get_all_loras()
        source_count = len(source_loras)
        target_count = self.target.count_loras()

        # Check for missing LoRAs
        missing_names = []
        for lora in source_loras:
            if self.target.get_lora_by_name(lora.name) is None:
                missing_names.append(lora.name)

        return source_count, target_count, missing_names

    def migrate_single(self, lora_name: str) -> bool:
        """
        Migrate a single LoRA by name.

        Args:
            lora_name: Name of LoRA to migrate

        Returns:
            True if migrated successfully, False otherwise
        """
        try:
            lora = self.source.get_lora_by_name(lora_name)
            if lora is None:
                logger.warning(f"LoRA not found in source: {lora_name}")
                return False

            self.target.add_lora(lora)
            logger.info(f"Successfully migrated: {lora_name}")
            return True

        except Exception as e:
            logger.error(f"Error migrating {lora_name}: {e}")
            return False


def create_migration_script(
    legacy_registry_path: Optional[Path] = None,
    sqlite_db_path: Optional[Path] = None,
) -> str:
    """
    Generate a migration script template.

    Args:
        legacy_registry_path: Path to legacy registry (optional)
        sqlite_db_path: Path to SQLite database (optional)

    Returns:
        Python script as string
    """
    registry_path = legacy_registry_path or Path("path/to/legacy/registry")
    db_path = sqlite_db_path or Path("loras.db")

    script = f'''"""
Migration script: Legacy ModelRegistry -> SQLiteModelRepository

This script migrates all LoRA data from the old ModelRegistry
to the new SQLiteModelRepository.

Generated automatically by RepositoryMigrationHelper.
"""

from pathlib import Path
from ml_lib.diffusion.infrastructure.persistence.sqlite_model_repository import (
    SQLiteModelRepository,
)
from ml_lib.diffusion.infrastructure.persistence.model_registry_adapter import (
    ModelRegistryAdapter,
)
from ml_lib.diffusion.infrastructure.persistence.migration_helper import (
    RepositoryMigrationHelper,
)
# TODO: Import your legacy ModelRegistry
# from ml_lib.diffusion.services.model_registry import ModelRegistry

import logging

logging.basicConfig(level=logging.INFO)


def main():
    """Run migration."""
    print("=" * 70)
    print("  MIGRATION: Legacy Registry -> SQLite Repository")
    print("=" * 70)
    print()

    # Step 1: Load legacy registry
    print("Step 1: Loading legacy ModelRegistry...")
    # TODO: Initialize your legacy ModelRegistry
    # legacy_registry = ModelRegistry.load("{registry_path}")
    # legacy_adapter = ModelRegistryAdapter(legacy_registry)
    print("  [!] TODO: Uncomment and configure legacy registry")
    print()

    # Step 2: Create SQLite repository
    print("Step 2: Creating SQLite repository...")
    sqlite_repo = SQLiteModelRepository(db_path=Path("{db_path}"))
    print(f"  ✅ SQLite database: {db_path}")
    print()

    # Step 3: Run migration
    print("Step 3: Running migration...")
    # TODO: Uncomment when legacy_adapter is ready
    # helper = RepositoryMigrationHelper(
    #     source=legacy_adapter,
    #     target=sqlite_repo,
    # )
    # migrated_count = helper.migrate_all(skip_existing=True)
    # print(f"  ✅ Migrated {{migrated_count}} LoRAs")
    print("  [!] TODO: Uncomment migration code")
    print()

    # Step 4: Verify migration
    print("Step 4: Verifying migration...")
    # TODO: Uncomment when ready
    # source_count, target_count, missing = helper.verify_migration()
    # print(f"  Source: {{source_count}} LoRAs")
    # print(f"  Target: {{target_count}} LoRAs")
    # if missing:
    #     print(f"  ⚠ Missing {{len(missing)}} LoRAs: {{missing}}")
    # else:
    #     print("  ✅ All LoRAs migrated successfully")
    print("  [!] TODO: Uncomment verification code")
    print()

    print("=" * 70)
    print("  MIGRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
'''

    return script


def export_migration_script(
    output_path: Path,
    legacy_registry_path: Optional[Path] = None,
    sqlite_db_path: Optional[Path] = None,
) -> None:
    """
    Export migration script to file.

    Args:
        output_path: Where to save the script
        legacy_registry_path: Path to legacy registry (optional)
        sqlite_db_path: Path to SQLite database (optional)
    """
    script = create_migration_script(legacy_registry_path, sqlite_db_path)

    with open(output_path, "w") as f:
        f.write(script)

    logger.info(f"Migration script exported to: {output_path}")
