"""
Delete one table's signature from Milvus (sql_table_signatures collection).
Use when cleaning up after a rejected/duplicate job so re-ingest doesn't hit duplicate key.

Usage: python scripts/delete_milvus_signature.py <table_name>
Example: python scripts/delete_milvus_signature.py ethanol_blending_with_petrol
"""
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from dotenv import load_dotenv
load_dotenv(root / ".env")

from app.core.milvus_manager import milvus_manager


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/delete_milvus_signature.py <table_name>")
        print("Example: python scripts/delete_milvus_signature.py ethanol_blending_with_petrol")
        sys.exit(1)
    table_name = sys.argv[1].strip()

    if not milvus_manager.connect():
        print("Could not connect to Milvus. Check MILVUS_* in .env and that Milvus is running.")
        sys.exit(1)
    try:
        if milvus_manager.delete_signature(table_name):
            print(f"Milvus signature deleted for: {table_name}")
        else:
            print(f"Failed to delete signature for: {table_name}")
            sys.exit(1)
    finally:
        milvus_manager.disconnect()


if __name__ == "__main__":
    main()
