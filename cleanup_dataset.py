"""
SilentSpeak AI - Cleanup old dataset
Run this before re-recording to start fresh.
"""

import os
import shutil
from config import DATA_DIR

def cleanup():
    print("=" * 60)
    print("SilentSpeak AI - Dataset Cleanup")
    print("=" * 60)

    if not os.path.exists(DATA_DIR):
        print(f"Data directory does not exist: {DATA_DIR}")
        return

    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]

    if not files:
        print("No CSV files found. Nothing to clean.")
        return

    print(f"\nFound {len(files)} CSV files:")
    for f in sorted(files):
        filepath = os.path.join(DATA_DIR, f)
        size = os.path.getsize(filepath)
        print(f"  {f:20s} ({size:6d} bytes)")

    print(f"\nWARNING: This will DELETE all {len(files)} files permanently!")
    confirm = input("Type 'DELETE' to confirm cleanup: ").strip()

    if confirm == "DELETE":
        backup_dir = os.path.join(os.path.dirname(DATA_DIR), "raw_backup")
        os.makedirs(backup_dir, exist_ok=True)

        for f in files:
            src = os.path.join(DATA_DIR, f)
            dst = os.path.join(backup_dir, f)
            shutil.move(src, dst)

        print(f"\nMoved {len(files)} files to {backup_dir}")
        print("Cleanup complete. Ready for fresh recording.")
    else:
        print("\nCleanup cancelled.")

if __name__ == "__main__":
    cleanup()