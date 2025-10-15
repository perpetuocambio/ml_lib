#!/usr/bin/env python3
"""
CLI tool to migrate ComfyUI metadata to SQLite database.

Usage:
    python scripts/migrate_metadata.py /path/to/ComfyUI
"""

import sys
import logging
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_lib.diffusion.models.metadata_db import MetadataDatabase
from ml_lib.diffusion.sources.comfyui_migrator import ComfyUIMetadataMigrator
from ml_lib.diffusion.config import detect_comfyui_installation

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def main():
    """Main migration function."""
    print("=" * 80)
    print("ComfyUI Metadata Migration to SQLite")
    print("=" * 80)
    print()

    # Detect or use provided ComfyUI path
    if len(sys.argv) > 1:
        comfyui_root = Path(sys.argv[1])
    else:
        print("Auto-detecting ComfyUI installation...")
        comfyui_root = detect_comfyui_installation()

    if not comfyui_root or not comfyui_root.exists():
        print("❌ ComfyUI installation not found!")
        print("Usage: python scripts/migrate_metadata.py /path/to/ComfyUI")
        sys.exit(1)

    print(f"✅ Found ComfyUI at: {comfyui_root}")
    print()

    # Initialize database
    db_path = Path(__file__).parent.parent / "data" / "models.db"
    print(f"📁 Database: {db_path}")

    db = MetadataDatabase(db_path)
    print("✅ Database initialized")
    print()

    # Create migrator
    migrator = ComfyUIMetadataMigrator(db)

    # Migrate
    print("🔄 Starting migration...")
    print()

    results = migrator.migrate_comfyui_installation(comfyui_root)

    # Report results
    print()
    print("=" * 80)
    print("Migration Results")
    print("=" * 80)

    total_success = 0
    total_failed = 0

    for model_type, (successful, failed) in results.items():
        total_success += successful
        total_failed += failed
        status = "✅" if failed == 0 else "⚠️"
        print(f"{status} {model_type}: {successful} successful, {failed} failed")

    print()
    print(f"Total: {total_success} successful, {total_failed} failed")
    print()

    # Show database stats
    stats = db.get_stats()
    print("=" * 80)
    print("Database Statistics")
    print("=" * 80)
    print(f"Total models: {stats.get('total', 0)}")
    print(f"Local models: {stats.get('local_models', 0)}")
    print()

    print("By type:")
    for model_type, count in stats.get("by_type", {}).items():
        print(f"  - {model_type}: {count}")

    print()
    print("By base model:")
    for base_model, count in stats.get("by_base_model", {}).items():
        print(f"  - {base_model}: {count}")

    print()
    print("=" * 80)
    print("✅ Migration complete!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Migration cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Migration failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
