"""
Clear all ingestion history and DB data for a completely clean state.

- Deletes all files in output/, uploads/, processed/
- Clears in-memory jobs directly (job_manager.clear_all(), no server needed)
- PostgreSQL: drops all ingestion tables, clears tables_metadata & operational_metadata
- Milvus: drops the signatures collection (no similar-table history → no incremental loads)

Requires .env with POSTGRES_* (and MILVUS_* if Milvus is used).

Usage:
  python clear_history.py              # full reset: files + jobs + Postgres + Milvus
  python clear_history.py --no-db      # clear only files and in-memory jobs
  python clear_history.py --dry-run    # show what would be done, no changes
"""
import argparse
import os
from pathlib import Path

# Load .env before importing app (for DB connection)
_env = Path(__file__).resolve().parent / ".env"
if _env.exists():
    with _env.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
UPLOAD_DIR = ROOT / "uploads"
PROCESSED_DIR = ROOT / "processed"


def clear_dirs(dry_run: bool) -> int:
    """Remove all files in output/, uploads/, processed/. Returns count deleted."""
    deleted = 0
    for d in (OUTPUT_DIR, UPLOAD_DIR, PROCESSED_DIR):
        if not d.exists():
            continue
        for f in d.iterdir():
            if f.is_file():
                if dry_run:
                    print(f"  [dry-run] would delete {f.name}")
                else:
                    f.unlink()
                    print(f"  Deleted {f}")
                deleted += 1
    return deleted


def clear_jobs_api(dry_run: bool) -> bool:
    """Clear in-memory jobs directly via job_manager (no server needed)."""
    if dry_run:
        print("  [dry-run] would clear in-memory jobs (job_manager.clear_all())")
        return True
    try:
        from app.core.job_manager import job_manager
        n = job_manager.clear_all()
        print(f"  Cleared {n} job(s) from job manager")
        return True
    except Exception as e:
        print(f"  Failed to clear jobs: {e}")
        return False


def clear_db(dry_run: bool) -> dict:
    """Drop all tables from tables_metadata and clear metadata tables. Returns {'tables_dropped': n}."""
    if dry_run:
        try:
            from app.core.database import db_manager
            tables = db_manager.list_tables_from_metadata()
            print(f"  [dry-run] would drop tables: {tables or '(none)'}")
            print("  [dry-run] would DELETE FROM tables_metadata, operational_metadata")
        except Exception as e:
            print(f"  [dry-run] db: {e}")
        return {"tables_dropped": 0}
    try:
        from app.core.database import db_manager
        result = db_manager.drop_ingestion_tables_and_metadata()
        print(f"  Dropped {result['tables_dropped']} table(s), cleared metadata")
        return result
    except Exception as e:
        print(f"  DB clear failed: {e}")
        return {"tables_dropped": 0}


def clear_milvus(dry_run: bool) -> bool:
    """Drop Milvus signatures collection so no similar-table / incremental-load history remains."""
    if dry_run:
        print("  [dry-run] would drop Milvus collection (sql_table_signatures)")
        return True
    try:
        from app.core.milvus_manager import milvus_manager
        if milvus_manager.clear_collection():
            print("  Milvus collection dropped")
            return True
        print("  Milvus not cleared (server down or not configured)")
        return False
    except Exception as e:
        print(f"  Milvus clear failed (server may be down): {e}")
        return False


def main():
    ap = argparse.ArgumentParser(
        description="Clear all ingestion history and DB data (files, jobs, tables, metadata)."
    )
    ap.add_argument(
        "--no-db",
        action="store_true",
        help="Do not touch DB; only clear output/uploads/processed and in-memory jobs",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be done; do not delete or change anything",
    )
    args = ap.parse_args()

    print("Clearing ingestion history and DB data (clean state)...")
    n_files = clear_dirs(args.dry_run)
    if not args.dry_run:
        print(f"  Removed {n_files} file(s) from output/uploads/processed")
    clear_jobs_api(args.dry_run)
    if not args.no_db:
        clear_db(args.dry_run)
        clear_milvus(args.dry_run)
    else:
        print("  Skipping DB and Milvus (--no-db)")
    print("Done. Start server and run_samples.py for a fresh run.")


if __name__ == "__main__":
    main()
