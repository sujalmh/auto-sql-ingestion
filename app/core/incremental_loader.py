"""
Incremental Loader — Generic Schema-Evolving UPSERT Pipeline
=============================================================

Responsibilities
----------------
1. Detect primary / unique key columns (from DB constraints or explicit config).
2. Compare incoming CSV schema with existing table schema.
3. Evolve the table schema automatically:
   - ALTER TABLE ADD COLUMN for genuinely new columns (nullable).
   - Widen column types safely (e.g. INTEGER → BIGINT → TEXT).
   - Never drop existing columns.
4. Align DataFrame columns to the (evolved) table schema by name
   (case-insensitive, trim).
5. Perform a deterministic UPSERT (INSERT … ON CONFLICT DO UPDATE).
6. Maintain the `ingested_at` audit timestamp on every row.
7. Return a rich audit summary.

Re-running the same file is fully idempotent.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
from datetime import datetime

from app.core.database import db_manager, _WIDENING_GROUPS
from app.core.logger import logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_col(col: str) -> str:
    """Lowercase + strip so all comparisons are case-insensitive."""
    return col.strip().lower()


def _infer_pg_type(series: pd.Series) -> str:
    """
    Cheap heuristic to infer a PostgreSQL column type from a pandas Series.
    Falls back to TEXT when uncertain — never breaks the pipeline.
    """
    non_null = series.dropna()
    if non_null.empty:
        return "TEXT"

    # Pandas dtype shortcuts
    dtype_str = str(series.dtype)
    if "int" in dtype_str:
        return "BIGINT"
    if "float" in dtype_str:
        return "DOUBLE PRECISION"
    if "bool" in dtype_str:
        return "BOOLEAN"
    if "datetime" in dtype_str:
        return "TIMESTAMP WITHOUT TIME ZONE"

    # Try numeric / date parsing on the string values
    sample = non_null.astype(str).head(50)
    # Integer?
    try:
        sample.astype(int)
        return "BIGINT"
    except (ValueError, TypeError):
        pass
    # Float?
    try:
        sample.astype(float)
        return "DOUBLE PRECISION"
    except (ValueError, TypeError):
        pass
    # Date/Timestamp?
    try:
        pd.to_datetime(sample, infer_datetime_format=True, errors="raise")
        return "TIMESTAMP WITHOUT TIME ZONE"
    except Exception:
        pass

    return "TEXT"


def _is_safe_widening(from_type: str, to_type: str) -> bool:
    """
    Return True if upgrading from_type → to_type is a safe, lossless widening
    according to the _WIDENING_GROUPS table in database.py.
    """
    from_norm = from_type.lower()
    to_norm   = to_type.lower()
    allowed = _WIDENING_GROUPS.get(from_norm, [])
    return to_norm in allowed


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class IncrementalLoader:
    """
    Handles incremental data loading with dynamic schema evolution.

    Usage
    -----
    summary = incremental_loader.perform_incremental_load(
        table_name="my_table",
        df=new_data_df,
        column_types=inferred_col_types,   # from LLM / preprocessor
        key_columns=["id", "period"],       # optional override; auto-detects from PK
    )
    """

    def __init__(self):
        logger.info("IncrementalLoader initialised (schema-evolution + upsert mode)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def perform_incremental_load(
        self,
        table_name: str,
        df: pd.DataFrame,
        column_types: Dict[str, str],
        key_columns: Optional[List[str]] = None,
        last_available_value: Optional[str] = None,
    ) -> Dict:
        """
        Complete incremental load with schema evolution and UPSERT.

        Steps
        -----
        1. Resolve key columns (PK from DB, then explicit config, then warn).
        2. Fetch existing table schema.
        3. Detect schema differences (new cols, missing cols, type changes).
        4. Evolve table schema (ALTER TABLE ADD COLUMN, safe type widening).
        5. Ensure `ingested_at` audit column exists.
        6. Align DataFrame to evolved table schema.
        7. UPSERT data.
        8. Update operational metadata.
        9. Return audit summary.

        Args:
            table_name:          Existing PostgreSQL table name.
            df:                  Preprocessed DataFrame from the pipeline.
            column_types:        LLM-inferred {col_name: pg_type} for the CSV.
            key_columns:         Explicit PK override; auto-detected when None.
            last_available_value: Optional period tracking value for metadata.

        Returns:
            Dictionary with keys:
                success, table_name, rows_before, rows_inserted, rows_updated,
                rows_after, columns_added, schema_changes, warnings, timestamp
        """
        logger.info(f"[IL] Starting incremental load for '{table_name}'")

        warnings_log: List[str] = []
        columns_added: List[str] = []
        schema_changes: List[str] = []
        rows_before = 0

        try:
            # ---------------------------------------------------------
            # Step 1: Get current row count (before)
            # ---------------------------------------------------------
            rows_before = self._get_row_count(table_name)
            logger.info(f"[IL] Rows before load: {rows_before}")

            # ---------------------------------------------------------
            # Step 2: Resolve primary / unique key columns
            # ---------------------------------------------------------
            effective_keys = self._resolve_keys(table_name, key_columns, warnings_log)

            # ---------------------------------------------------------
            # Step 3: Fetch existing table schema
            # ---------------------------------------------------------
            existing_schema: Dict[str, str] = db_manager.get_table_column_types(table_name)
            logger.info(
                f"[IL] Existing schema: {len(existing_schema)} columns — "
                f"{list(existing_schema.keys())[:10]}"
            )

            # ---------------------------------------------------------
            # Step 4: Detect schema differences
            # ---------------------------------------------------------
            new_cols, type_mismatches = self._detect_schema_diff(
                df, column_types, existing_schema, warnings_log
            )

            # ---------------------------------------------------------
            # Step 5: Evolve schema
            # ---------------------------------------------------------
            if new_cols:
                added = db_manager.alter_table_add_columns(table_name, new_cols)
                columns_added.extend(added)
                for col in added:
                    schema_changes.append(f"ADD COLUMN {col} {new_cols[col]}")
                not_added = [c for c in new_cols if c not in added]
                if not_added:
                    warnings_log.append(f"Could not add columns (see DB log): {not_added}")

            # Handle type widenings
            for col, info in type_mismatches.items():
                if info["safe"]:
                    ok = db_manager.alter_column_type_safe(
                        table_name, col, info["to"]
                    )
                    if ok:
                        schema_changes.append(
                            f"WIDEN COLUMN {col}: {info['from']} → {info['to']}"
                        )
                    else:
                        warnings_log.append(
                            f"Type widening failed for '{col}' "
                            f"({info['from']} → {info['to']}); data cast may fail"
                        )
                else:
                    warnings_log.append(
                        f"UNSAFE type change for '{col}': "
                        f"{info['from']} → {info['to']} (skipped; data will be cast to TEXT)"
                    )

            # ---------------------------------------------------------
            # Step 6: Ensure ingested_at audit column
            # ---------------------------------------------------------
            db_manager.ensure_ingested_at_column(table_name)

            # Re-fetch the evolved schema so alignment is accurate
            evolved_schema = db_manager.get_table_column_types(table_name)

            # ---------------------------------------------------------
            # Step 7: Align DataFrame to evolved table schema
            # ---------------------------------------------------------
            aligned_df = self._align_dataframe(df, evolved_schema, warnings_log)

            if aligned_df.empty and not df.empty:
                raise ValueError(
                    "Column alignment produced an empty DataFrame — "
                    "no columns match the table schema."
                )

            # ---------------------------------------------------------
            # Step 8: UPSERT
            # ---------------------------------------------------------
            if effective_keys:
                # Filter key_columns to only those actually present in aligned_df
                valid_keys = [
                    k for k in effective_keys
                    if _normalise_col(k) in [_normalise_col(c) for c in aligned_df.columns]
                ]
                if len(valid_keys) < len(effective_keys):
                    missing_keys = set(effective_keys) - set(valid_keys)
                    warnings_log.append(
                        f"Key column(s) not in aligned DataFrame, skipping: {missing_keys}"
                    )
                if valid_keys:
                    rows_inserted, rows_updated = db_manager.upsert_data(
                        table_name=table_name,
                        df=aligned_df,
                        key_columns=valid_keys,
                        column_types=column_types,
                    )
                else:
                    # Fallback — all resolved keys are missing from the file
                    warnings_log.append(
                        "No valid key columns found in data; falling back to append-only."
                    )
                    rows_inserted = db_manager.insert_data(
                        table_name, aligned_df, column_types=column_types
                    )
                    rows_updated = 0
            else:
                # No PK defined — append only
                rows_inserted = db_manager.insert_data(
                    table_name, aligned_df, column_types=column_types
                )
                rows_updated = 0

            logger.info(
                f"[IL] UPSERT complete — inserted: {rows_inserted}, updated: {rows_updated}"
            )

            # ---------------------------------------------------------
            # Step 9: Update operational metadata
            # ---------------------------------------------------------
            self._update_operational_metadata(
                table_name,
                rows_inserted + rows_updated,
                last_available_value
            )

            rows_after = self._get_row_count(table_name) or (
                rows_before + rows_inserted
            )

            if schema_changes:
                logger.info(f"[IL] Schema changes applied: {schema_changes}")
            if columns_added:
                logger.info(f"[IL] New columns added to table: {columns_added}")
            if warnings_log:
                for w in warnings_log:
                    logger.warning(f"[IL] {w}")

            return {
                "success": True,
                "table_name": table_name,
                "rows_before": rows_before,
                "rows_inserted": rows_inserted,
                "rows_updated": rows_updated,
                "rows_added": rows_inserted,   # legacy field
                "rows_after": rows_after,
                "columns_added": columns_added,
                "schema_changes": schema_changes,
                "warnings": warnings_log,
                "key_columns_used": effective_keys,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"[IL] Incremental load failed for '{table_name}': {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "table_name": table_name,
                "rows_before": rows_before,
                "rows_inserted": 0,
                "rows_updated": 0,
                "rows_added": 0,
                "rows_after": rows_before,
                "columns_added": columns_added,
                "schema_changes": schema_changes,
                "warnings": warnings_log,
            }

    # ------------------------------------------------------------------
    # Legacy helpers (kept for backward compatibility with other callers)
    # ------------------------------------------------------------------

    def get_table_columns(self, table_name: str) -> Optional[List[str]]:
        """Return ordered column names for an existing table."""
        schema = db_manager.get_table_column_types(table_name)
        return list(schema.keys()) if schema else None

    def get_current_row_count(self, table_name: str) -> Optional[int]:
        """
        Get current row count from operational_metadata.
        Falls back to COUNT(*) on the table itself if metadata is absent.
        """
        return self._get_row_count(table_name)

    def append_data(
        self,
        table_name: str,
        df: pd.DataFrame,
        column_types: Dict[str, str],
    ) -> bool:
        """
        Append-only insertion (no upsert).  Kept for backward compatibility.
        Prefer perform_incremental_load() for production use.
        """
        try:
            existing_cols = self.get_table_columns(table_name)
            if not existing_cols:
                return False
            aligned_df = self._simple_align(df, existing_cols)
            rows = db_manager.insert_data(table_name, aligned_df, column_types=column_types)
            return rows > 0
        except Exception as e:
            logger.error(f"append_data failed for '{table_name}': {e}")
            return False

    def update_operational_metadata(
        self,
        table_name: str,
        rows_added: int,
        last_available_value: Optional[str] = None,
    ) -> bool:
        """Public wrapper for the metadata update helper."""
        return self._update_operational_metadata(table_name, rows_added, last_available_value)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_row_count(self, table_name: str) -> int:
        """Return the actual row count directly from the table (COUNT(*))."""
        try:
            with db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    from psycopg2 import sql as psql
                    cur.execute(
                        psql.SQL("SELECT COUNT(*) FROM {}").format(
                            psql.Identifier(table_name.lower())
                        )
                    )
                    result = cur.fetchone()
                    return result[0] if result else 0
        except Exception as e:
            logger.warning(f"Could not fetch row count for '{table_name}': {e}")
            return 0

    def _resolve_keys(
        self,
        table_name: str,
        explicit_keys: Optional[List[str]],
        warnings: List[str],
    ) -> List[str]:
        """
        Resolve the effective key columns for UPSERT conflict resolution.

        Priority:
          1. Primary key from DB constraints (authoritative).
          2. Explicit key_columns config parameter.
          3. None (append-only fallback, with warning).
        """
        # Always try DB first
        db_pks = db_manager.get_primary_keys(table_name)
        if db_pks:
            logger.info(f"[IL] Using PK from DB constraints: {db_pks}")
            return db_pks

        if explicit_keys:
            logger.info(f"[IL] No DB PK found; using explicit key_columns: {explicit_keys}")
            return [k.strip().lower() for k in explicit_keys]

        warnings.append(
            f"No primary key found on '{table_name}' and no key_columns "
            "provided — falling back to append-only insert (no deduplication)."
        )
        logger.warning(
            f"[IL] No PK or explicit keys for '{table_name}'; "
            "using append-only mode (data may duplicate on re-run)."
        )
        return []

    def _detect_schema_diff(
        self,
        df: pd.DataFrame,
        column_types: Dict[str, str],
        existing_schema: Dict[str, str],   # {col_lower: pg_type}
        warnings: List[str],
    ) -> Tuple[Dict[str, str], Dict[str, Dict]]:
        """
        Compare incoming CSV columns against existing table columns.

        Returns:
            new_cols:        {col_name: pg_type} — columns to ADD to the table.
            type_mismatches: {col_name: {from, to, safe}} — columns to WIDEN.
        """
        new_cols: Dict[str, str] = {}
        type_mismatches: Dict[str, Dict] = {}

        # Build a normalised map of CSV columns
        csv_norm: Dict[str, str] = {_normalise_col(c): c for c in df.columns}
        # Normalised column_types map
        ct_norm: Dict[str, str] = {
            _normalise_col(k): v for k, v in column_types.items()
        }

        for csv_norm_col, original_col in csv_norm.items():
            if csv_norm_col in ("ingested_at",):
                continue  # reserved pipeline column

            if csv_norm_col not in existing_schema:
                # This column is new — schedule an ALTER TABLE
                inferred_type = ct_norm.get(
                    csv_norm_col,
                    _infer_pg_type(df[original_col])
                )
                new_cols[csv_norm_col] = inferred_type
                logger.info(
                    f"[IL] New column detected: '{csv_norm_col}' ({inferred_type})"
                )
            else:
                # Column exists — check for type drift
                existing_type = existing_schema[csv_norm_col].lower()
                incoming_type = ct_norm.get(csv_norm_col, "text").lower()

                # Normalise common aliases
                incoming_type = self._normalise_pg_type(incoming_type)
                existing_type  = self._normalise_pg_type(existing_type)

                if incoming_type != existing_type and incoming_type != "text":
                    safe = _is_safe_widening(existing_type, incoming_type)
                    type_mismatches[csv_norm_col] = {
                        "from": existing_type,
                        "to":   incoming_type,
                        "safe": safe,
                    }
                    level = "INFO" if safe else "WARNING"
                    getattr(logger, level.lower())(
                        f"[IL] Type {'widening' if safe else 'MISMATCH (unsafe)'} "
                        f"for '{csv_norm_col}': {existing_type} → {incoming_type}"
                    )
                    if not safe:
                        warnings.append(
                            f"Unsafe type change for '{csv_norm_col}': "
                            f"{existing_type} → {incoming_type}. "
                            "Column will be cast to TEXT to preserve data."
                        )

        # Report columns in the table but absent from the CSV (informational only)
        missing_from_csv = [
            c for c in existing_schema
            if c not in csv_norm and c != "ingested_at"
        ]
        if missing_from_csv:
            logger.info(
                f"[IL] Columns in table but absent from CSV "
                f"(will be NULL for new rows): {missing_from_csv}"
            )

        return new_cols, type_mismatches

    def _align_dataframe(
        self,
        df: pd.DataFrame,
        evolved_schema: Dict[str, str],  # {col_lower: pg_type}
        warnings: List[str],
    ) -> pd.DataFrame:
        """
        Produce an aligned DataFrame whose columns:
          - Match the evolved table schema by name (case-insensitive).
          - Are in the table's column order.
          - Have NULL for any table column missing from the CSV.
          - Exclude the pipeline-managed `ingested_at` column (added by upsert_data).
          - Exclude any CSV columns that are not in the table schema.

        Also handles the unsafe-type-change case by casting the relevant columns
        to TEXT so at least the data is preserved.
        """
        df_col_map: Dict[str, str] = {_normalise_col(c): c for c in df.columns}

        aligned_df = pd.DataFrame()
        for tbl_col in evolved_schema:
            if tbl_col == "ingested_at":
                continue  # managed by upsert_data

            if tbl_col in df_col_map:
                original_name = df_col_map[tbl_col]
                aligned_df[tbl_col] = df[original_name].values
            else:
                # Column in table but not in CSV — fill with NULL
                aligned_df[tbl_col] = None

        # Any CSV columns not in the evolved schema are silently dropped
        extra_csv_cols = [
            c for nc, c in df_col_map.items()
            if nc not in evolved_schema and nc != "ingested_at"
        ]
        if extra_csv_cols:
            logger.debug(
                f"[IL] CSV columns dropped (not in table schema): {extra_csv_cols}"
            )

        logger.info(f"[IL] Aligned DataFrame: {len(aligned_df.columns)} columns matched")
        return aligned_df

    def _simple_align(
        self,
        df: pd.DataFrame,
        existing_cols: List[str],
    ) -> pd.DataFrame:
        """Simple alignment for the legacy append_data method."""
        df_col_map = {_normalise_col(c): c for c in df.columns}
        aligned = pd.DataFrame()
        for col in existing_cols:
            nc = _normalise_col(col)
            if nc in df_col_map:
                aligned[col] = df[df_col_map[nc]].values
            else:
                aligned[col] = None
        return aligned

    @staticmethod
    def _normalise_pg_type(pg_type: str) -> str:
        """
        Collapse common PostgreSQL type aliases to a canonical form for
        clean comparisons.  E.g. 'int4' → 'integer', 'varchar' → 'character varying'.
        """
        aliases = {
            "int":          "integer",
            "int4":         "integer",
            "int8":         "bigint",
            "int2":         "smallint",
            "serial":       "integer",
            "bigserial":    "bigint",
            "float4":       "real",
            "float8":       "double precision",
            "float":        "double precision",
            "varchar":      "character varying",
            "char":         "character",
            "bool":         "boolean",
            "decimal":      "numeric",
            "timestamp":    "timestamp without time zone",
            "timestamptz":  "timestamp with time zone",
        }
        return aliases.get(pg_type.lower(), pg_type.lower())

    def _update_operational_metadata(
        self,
        table_name: str,
        rows_added: int,
        last_available_value: Optional[str] = None,
    ) -> bool:
        """Update operational_metadata after a successful incremental load."""
        try:
            with db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    if last_available_value:
                        cur.execute(
                            """
                            UPDATE operational_metadata
                            SET rows_count          = rows_count + %s,
                                last_updated_on     = %s,
                                last_available_value = %s
                            WHERE table_name = %s
                            """,
                            (rows_added, datetime.now(), last_available_value, table_name.lower()),
                        )
                    else:
                        cur.execute(
                            """
                            UPDATE operational_metadata
                            SET rows_count      = rows_count + %s,
                                last_updated_on = %s
                            WHERE table_name = %s
                            """,
                            (rows_added, datetime.now(), table_name.lower()),
                        )
                    conn.commit()
                    if cur.rowcount > 0:
                        logger.info(
                            f"[IL] Operational metadata updated for '{table_name}': "
                            f"+{rows_added} rows"
                        )
                        return True
                    else:
                        logger.warning(
                            f"[IL] No metadata row for '{table_name}' — metadata not updated"
                        )
                        return False
        except Exception as e:
            logger.error(f"[IL] Error updating operational metadata for '{table_name}': {e}")
            return False


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------
incremental_loader = IncrementalLoader()
