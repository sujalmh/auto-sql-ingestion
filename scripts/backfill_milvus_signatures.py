"""
Backfill Milvus signatures for existing PostgreSQL tables.

Problem this solves:
- New ingested tables get signatures during ingestion.
- Older existing tables may not have signatures in Milvus.

This script scans existing tables, detects which ones are missing signatures,
and creates signatures + embeddings for those tables only.

Usage:
  python scripts/backfill_milvus_signatures.py
  python scripts/backfill_milvus_signatures.py --all-public
  python scripts/backfill_milvus_signatures.py --tables table_a,table_b
  python scripts/backfill_milvus_signatures.py --dry-run
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from dotenv import load_dotenv
from psycopg2 import sql

from app.core.database import db_manager
from app.core.logger import logger
from app.core.milvus_manager import milvus_manager
from app.core.signature_builder import signature_builder

load_dotenv(root / ".env")


EXCLUDED_TABLES = {
    "tables_metadata",
    "operational_metadata",
}


def list_public_tables() -> List[str]:
    """Return all public tables except internal metadata tables."""
    try:
        with db_manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                      AND table_type = 'BASE TABLE'
                    ORDER BY table_name
                    """
                )
                tables = [row[0] for row in cur.fetchall()]
        return [t for t in tables if t.lower() not in EXCLUDED_TABLES]
    except Exception as exc:
        logger.error(f"Failed to list public tables: {exc}")
        return []


def resolve_target_tables(explicit_tables: Optional[str], all_public: bool) -> List[str]:
    """Resolve target table list from CLI flags."""
    if explicit_tables:
        return [t.strip() for t in explicit_tables.split(",") if t.strip()]

    if all_public:
        return list_public_tables()

    # Default: ingestion-managed tables from metadata.
    tables = db_manager.list_tables_from_metadata()
    if tables:
        return sorted(set(tables))

    # Fallback if metadata is empty.
    return list_public_tables()


def has_milvus_signature(table_name: str) -> bool:
    """Check whether a table already has a signature row in Milvus."""
    escaped = table_name.replace('"', '\\"')
    expr = f'table_name == "{escaped}"'

    try:
        rows = milvus_manager.collection.query(
            expr=expr,
            output_fields=["table_name"],
            limit=1,
        )
        return len(rows) > 0
    except Exception as exc:
        logger.warning(f"Could not query signature for '{table_name}': {exc}")
        return False


def fetch_sample_rows(table_name: str, sample_size: int = 5) -> pd.DataFrame:
    """Fetch top N rows for signature sample."""
    query = sql.SQL("SELECT * FROM {} LIMIT %s").format(sql.Identifier(table_name))

    with db_manager.get_connection() as conn:
        return pd.read_sql(query.as_string(conn), conn, params=[sample_size])


def fetch_row_count(table_name: str) -> int:
    """Fetch table row count."""
    query = sql.SQL("SELECT COUNT(*) FROM {}") .format(sql.Identifier(table_name))

    with db_manager.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            return int(cur.fetchone()[0])


def build_signature(table_name: str) -> Tuple[Dict, List[float]]:
    """
    Build signature + embedding for an existing table.

    Uses:
    - Actual row_count from the table.
    - First 5 rows for sample_rows.
    - information_schema-driven column types.
    """
    sample_df = fetch_sample_rows(table_name, sample_size=5)
    row_count = fetch_row_count(table_name)

    if sample_df is None:
        sample_df = pd.DataFrame()

    column_types = db_manager.get_table_column_types(table_name)

    # Ensure every sampled column has a type fallback.
    for col in sample_df.columns:
        column_types.setdefault(col, "text")

    sample_rows = sample_df.to_dict(orient="records")

    signature = {
        "table_name": table_name,
        "columns": sample_df.columns.tolist(),
        "column_types": column_types,
        "sample_rows": sample_rows,
        "row_count": row_count,
    }

    embedding = signature_builder.create_embedding(signature)
    return signature, embedding


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill missing Milvus signatures")
    parser.add_argument(
        "--tables",
        type=str,
        default=None,
        help="Comma-separated table names to process",
    )
    parser.add_argument(
        "--all-public",
        action="store_true",
        help="Process all public tables instead of metadata-listed tables",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions only, do not create signatures",
    )

    args = parser.parse_args()

    tables = resolve_target_tables(args.tables, args.all_public)
    if not tables:
        print("No tables found to process.")
        return

    print(f"Discovered {len(tables)} table(s) to evaluate")

    if not milvus_manager.connect():
        print("Could not connect to Milvus. Check MILVUS_* settings and service status.")
        sys.exit(1)

    try:
        if not milvus_manager.create_collection():
            print("Could not initialize Milvus collection.")
            sys.exit(1)

        existing: Set[str] = set()
        missing: List[str] = []

        for table in tables:
            if has_milvus_signature(table):
                existing.add(table)
            else:
                missing.append(table)

        print(f"Already signed: {len(existing)}")
        print(f"Missing signatures: {len(missing)}")

        if args.dry_run:
            if missing:
                print("\nWould create signatures for:")
                for table in missing:
                    print(f"  - {table}")
            return

        created = 0
        failed = 0

        for idx, table in enumerate(missing, start=1):
            try:
                print(f"[{idx}/{len(missing)}] Creating signature: {table}")
                signature, embedding = build_signature(table)
                ok = milvus_manager.insert_signature(table, embedding, signature)
                if ok:
                    created += 1
                else:
                    failed += 1
                    print(f"  Failed to insert signature for: {table}")
            except Exception as exc:
                failed += 1
                logger.error(f"Backfill failed for '{table}': {exc}")
                print(f"  Error for {table}: {exc}")

        print("\nBackfill complete")
        print(f"Created: {created}")
        print(f"Failed: {failed}")
        print(f"Skipped (already present): {len(existing)}")

    finally:
        milvus_manager.disconnect()


if __name__ == "__main__":
    main()
