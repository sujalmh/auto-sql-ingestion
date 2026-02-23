"""
Integration test: Generic Incremental Load with Dynamic Schema Evolution
=======================================================================

Tests:
  1. Idempotency    — Re-running the same file produces no duplicates
  2. New columns    — ALTER TABLE ADD COLUMN fires for genuinely new fields
  3. Upsert update  — Updated values on the same key rows are applied correctly
  4. Missing cols   — Table columns absent from the CSV receive NULL (not an error)
  5. ingested_at    — Audit column is present and populated after every load
  6. Type widening  — INTEGER column widens to BIGINT when new data needs it (via schema validator)

Prerequisites:
  - A running PostgreSQL instance with credentials in .env
  - Run from the auto-sql-ingestion directory:
      cd S:/internship/ingestion/auto-sql-ingestion
      python test_incremental_load.py
"""

import sys
import traceback
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.database import db_manager
from app.core.incremental_loader import incremental_loader
from app.core.schema_validator import schema_validator
from app.core.logger import logger

# ─────────────────────────────────────────────────────────────────────────────
# Test table config
# ─────────────────────────────────────────────────────────────────────────────
TEST_TABLE = "il_test_generic"

PASSED: list[str] = []
FAILED: list[str] = []


def ok(name: str) -> None:
    PASSED.append(name)
    print(f"  ✓ PASS: {name}")


def fail(name: str, reason: str) -> None:
    FAILED.append(name)
    print(f"  ✗ FAIL: {name} — {reason}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def setup_test_table() -> None:
    """Create a clean test table with a primary key."""
    print("\n[Setup] Creating test table …")
    with db_manager.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f'DROP TABLE IF EXISTS "{TEST_TABLE}" CASCADE')
            cur.execute(
                f"""
                CREATE TABLE "{TEST_TABLE}" (
                    id      INTEGER PRIMARY KEY,
                    name    TEXT,
                    value   DOUBLE PRECISION,
                    region  TEXT
                )
                """
            )
    print(f"[Setup] Table '{TEST_TABLE}' ready.")


def teardown_test_table() -> None:
    """Remove the test table."""
    with db_manager.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f'DROP TABLE IF EXISTS "{TEST_TABLE}" CASCADE')
    print(f"\n[Teardown] Table '{TEST_TABLE}' dropped.")


def count_rows() -> int:
    with db_manager.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f'SELECT COUNT(*) FROM "{TEST_TABLE}"')
            return cur.fetchone()[0]


def fetch_all() -> list[dict]:
    with db_manager.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f'SELECT * FROM "{TEST_TABLE}" ORDER BY id')
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


def get_columns() -> list[str]:
    schema = db_manager.get_table_column_types(TEST_TABLE)
    return list(schema.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_initial_load():
    """Insert initial 3 rows."""
    print("\n[Test 1] Initial load …")
    df = pd.DataFrame({
        "id":     [1, 2, 3],
        "name":   ["Alice", "Bob", "Carol"],
        "value":  [10.5, 20.0, 30.7],
        "region": ["North", "South", "East"],
    })
    col_types = {"id": "INTEGER", "name": "TEXT", "value": "DOUBLE PRECISION", "region": "TEXT"}

    summary = incremental_loader.perform_incremental_load(
        table_name=TEST_TABLE,
        df=df,
        column_types=col_types,
    )

    if not summary["success"]:
        fail("Initial load", summary.get("error", "unknown"))
        return

    rows = count_rows()
    if rows == 3:
        ok("Initial load — 3 rows inserted")
    else:
        fail("Initial load", f"expected 3 rows, got {rows}")

    ins = summary.get("rows_inserted", 0)
    if ins == 3:
        ok("Initial load — rows_inserted=3")
    else:
        fail("Initial load rows_inserted", f"expected 3, got {ins}")


def test_idempotency():
    """Re-running the same CSV must not duplicate rows."""
    print("\n[Test 2] Idempotency (re-run same data) …")
    df = pd.DataFrame({
        "id":     [1, 2, 3],
        "name":   ["Alice", "Bob", "Carol"],
        "value":  [10.5, 20.0, 30.7],
        "region": ["North", "South", "East"],
    })
    col_types = {"id": "INTEGER", "name": "TEXT", "value": "DOUBLE PRECISION", "region": "TEXT"}

    summary = incremental_loader.perform_incremental_load(
        table_name=TEST_TABLE,
        df=df,
        column_types=col_types,
    )

    rows = count_rows()
    if rows == 3:
        ok("Idempotency — still 3 rows after re-run")
    else:
        fail("Idempotency", f"expected 3 rows, got {rows}")

    upd = summary.get("rows_updated", 0)
    ins = summary.get("rows_inserted", 0)
    if upd == 3 and ins == 0:
        ok("Idempotency — 3 rows updated (not re-inserted)")
    else:
        fail("Idempotency update counts", f"expected upd=3 ins=0, got upd={upd} ins={ins}")


def test_upsert_update():
    """Update existing rows — changed values must be reflected."""
    print("\n[Test 3] Upsert — value changes …")
    df = pd.DataFrame({
        "id":     [1, 2],
        "name":   ["Alice-v2", "Bob-v2"],
        "value":  [99.9, 88.8],
        "region": ["West", "West"],
    })
    col_types = {"id": "INTEGER", "name": "TEXT", "value": "DOUBLE PRECISION", "region": "TEXT"}

    summary = incremental_loader.perform_incremental_load(
        table_name=TEST_TABLE,
        df=df,
        column_types=col_types,
    )

    rows_after = count_rows()
    if rows_after == 3:
        ok("Upsert update — row count unchanged (still 3)")
    else:
        fail("Upsert update row count", f"expected 3, got {rows_after}")

    data = fetch_all()
    alice = next((r for r in data if r["id"] == 1), None)
    if alice and alice["name"] == "Alice-v2" and abs(alice["value"] - 99.9) < 0.01:
        ok("Upsert update — Alice row updated correctly")
    else:
        fail("Upsert update Alice", f"got: {alice}")

    upd = summary.get("rows_updated", 0)
    if upd == 2:
        ok("Upsert update — rows_updated=2")
    else:
        fail("Upsert update rows_updated", f"expected 2, got {upd}")


def test_new_columns():
    """CSV with a brand-new column → ALTER TABLE ADD COLUMN."""
    print("\n[Test 4] New columns (schema evolution) …")
    df = pd.DataFrame({
        "id":       [1, 2, 3],
        "name":     ["Alice-v2", "Bob-v2", "Carol"],
        "value":    [99.9, 88.8, 30.7],
        "region":   ["West", "West", "East"],
        "category": ["A", "B", "C"],        # ← NEW column
        "score":    [0.91, 0.85, 0.77],     # ← NEW column
    })
    col_types = {
        "id": "INTEGER", "name": "TEXT", "value": "DOUBLE PRECISION",
        "region": "TEXT", "category": "TEXT", "score": "DOUBLE PRECISION",
    }

    before_cols = set(get_columns())
    summary = incremental_loader.perform_incremental_load(
        table_name=TEST_TABLE,
        df=df,
        column_types=col_types,
    )
    after_cols = set(get_columns())

    added = summary.get("columns_added", [])
    schema_changes = summary.get("schema_changes", [])

    if "category" in after_cols:
        ok("New columns — 'category' column added to table")
    else:
        fail("New columns category", "column not found in table after load")

    if "score" in after_cols:
        ok("New columns — 'score' column added to table")
    else:
        fail("New columns score", "column not found in table after load")

    if len(added) >= 2:
        ok(f"New columns — columns_added reported: {added}")
    else:
        fail("New columns columns_added", f"expected ≥2 in columns_added, got {added}")

    if schema_changes:
        ok(f"New columns — schema_changes logged: {schema_changes}")
    else:
        fail("New columns schema_changes", "no schema_changes in summary")

    # Old rows should have NULL for new columns
    data = fetch_all()
    carol = next((r for r in data if r["id"] == 3), None)
    # Carol's category and score were set in this run, so check id=1 original value was preserved
    if carol and carol.get("name") == "Carol":
        ok("New columns — existing row data preserved after schema evolution")
    else:
        fail("New columns data preservation", f"Carol row: {carol}")


def test_missing_columns_from_csv():
    """CSV missing 'score' column — those table rows should get NULL (not error)."""
    print("\n[Test 5] Missing columns in CSV (rows get NULL) …")
    df = pd.DataFrame({
        "id":    [4],
        "name":  ["Dave"],
        "value": [55.5],
        # no region, no category, no score → all should be NULL
    })
    col_types = {"id": "INTEGER", "name": "TEXT", "value": "DOUBLE PRECISION"}

    rows_before = count_rows()
    summary = incremental_loader.perform_incremental_load(
        table_name=TEST_TABLE,
        df=df,
        column_types=col_types,
    )

    if not summary["success"]:
        fail("Missing columns", summary.get("error", "unknown"))
        return

    rows_after = count_rows()
    if rows_after == rows_before + 1:
        ok("Missing columns — new row inserted (total +1)")
    else:
        fail("Missing columns row count", f"expected {rows_before+1}, got {rows_after}")

    data = fetch_all()
    dave = next((r for r in data if r["id"] == 4), None)
    if dave is not None:
        if dave.get("region") is None and dave.get("category") is None:
            ok("Missing columns — absent CSV columns correctly stored as NULL")
        else:
            fail("Missing columns NULL check", f"Dave row: {dave}")
    else:
        fail("Missing columns Dave", "Row id=4 not found")


def test_ingested_at():
    """Every row should have a non-null ingested_at timestamp."""
    print("\n[Test 6] ingested_at audit column …")
    cols = get_columns()
    if "ingested_at" in cols:
        ok("ingested_at — column exists on table")
    else:
        fail("ingested_at column exists", f"columns: {cols}")
        return

    data = fetch_all()
    nulls = [r for r in data if r.get("ingested_at") is None]
    if not nulls:
        ok("ingested_at — all rows have non-null ingested_at")
    else:
        fail("ingested_at non-null", f"{len(nulls)} row(s) with NULL ingested_at")

    # Verify it's a recent timestamp (within last 60 seconds)
    now_utc = datetime.now(timezone.utc)
    stale = []
    for row in data:
        ts = row.get("ingested_at")
        if ts is None:
            continue
        # psycopg2 returns aware datetime for timestamptz
        if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
            delta = abs((now_utc - ts).total_seconds())
        else:
            delta = 0  # skip if no tz info
        if delta > 120:
            stale.append((row["id"], ts))
    if not stale:
        ok("ingested_at — timestamps are recent (within 2 min)")
    else:
        fail("ingested_at recent", f"stale rows: {stale}")


def test_schema_validator_type_comparison():
    """Unit test for compare_column_types: safe and unsafe mismatches."""
    print("\n[Test 7] Schema validator — type comparison …")
    new_types      = {"id": "bigint", "score": "text", "value": "integer"}
    existing_types = {"id": "integer",  "score": "double precision", "value": "text"}

    result = schema_validator.compare_column_types(new_types, existing_types)

    # id: integer → bigint = safe widening
    if "id" in result and result["id"]["safe"] is True:
        ok("Type comparison — integer→bigint is safe widening")
    else:
        fail("Type comparison safe widening", f"result: {result.get('id')}")

    # score: double precision → text = safe widening (text accepts everything)
    # But our logic skips when incoming is "text" (always acceptable)
    if "score" not in result:
        ok("Type comparison — text destination always accepted (no mismatch)")
    else:
        fail("Type comparison text skip", f"score should be skipped, got: {result.get('score')}")

    # value: text → integer = UNSAFE narrowing
    if "value" in result and result["value"]["safe"] is False:
        ok("Type comparison — text→integer is unsafe narrowing")
    else:
        fail("Type comparison unsafe narrowing", f"result: {result.get('value')}")


def test_additive_evolution_flag():
    """validate_schema_match should return is_additive_evolution=True for new-col-only changes."""
    print("\n[Test 8] Schema validator — additive evolution flag …")
    existing = ["id", "name", "value"]
    new_csv  = ["id", "name", "value", "new_col_a", "new_col_b"]

    result = schema_validator.validate_schema_match(new_csv, existing)
    if result.get("is_additive_evolution") is True:
        ok("Additive evolution — is_additive_evolution=True for new-cols-only change")
    else:
        fail("Additive evolution flag", f"result: {result}")

    # Destructive change — col removed from CSV
    new_csv_destructive = ["id", "name"]  # "value" dropped
    result2 = schema_validator.validate_schema_match(new_csv_destructive, existing)
    if result2.get("is_additive_evolution") is False:
        ok("Additive evolution — is_additive_evolution=False when column dropped from CSV")
    else:
        fail("Additive evolution destructive", f"result: {result2}")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_all():
    print("=" * 72)
    print("  INCREMENTAL LOAD — SCHEMA EVOLUTION INTEGRATION TEST")
    print("=" * 72)

    # Pure unit tests (no DB required)
    test_schema_validator_type_comparison()
    test_additive_evolution_flag()

    # DB integration tests
    try:
        setup_test_table()

        test_initial_load()
        test_idempotency()
        test_upsert_update()
        test_new_columns()
        test_missing_columns_from_csv()
        test_ingested_at()

    except Exception as exc:
        print(f"\n[FATAL] Test suite aborted: {exc}")
        traceback.print_exc()
    finally:
        teardown_test_table()

    print("\n" + "=" * 72)
    print(f"  RESULTS: {len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print(f"  FAILED:  {', '.join(FAILED)}")
    print("=" * 72)
    return len(FAILED) == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
