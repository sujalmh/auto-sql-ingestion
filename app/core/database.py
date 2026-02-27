import re
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_batch, execute_values
import pandas as pd
from typing import Dict, Optional, List, Tuple
from contextlib import contextmanager
from datetime import datetime
from app.config import settings
from app.core.logger import logger

# Safe type widening order: earlier index = narrower type.
# A column can only be widened (index moves right), never narrowed automatically.
_TYPE_WIDENING_ORDER = [
    "smallint", "integer", "bigint", "numeric", "real", "double precision",
    "date", "timestamp without time zone", "timestamp with time zone",
    "character varying", "text"
]

_WIDENING_GROUPS: Dict[str, List[str]] = {
    # Integers widen to bigger integers / numeric
    "smallint": ["integer", "bigint", "numeric", "text"],
    "integer": ["bigint", "numeric", "text"],
    "bigint": ["numeric", "text"],
    "numeric": ["text"],
    "real": ["double precision", "numeric", "text"],
    "double precision": ["numeric", "text"],
    # Date/time hierarchy
    "date": ["timestamp without time zone", "timestamp with time zone", "text"],
    "timestamp without time zone": ["timestamp with time zone", "text"],
    "timestamp with time zone": ["text"],
    # Varchar / char always widen to text
    "character varying": ["text"],
    "character": ["text"],
    # Boolean can widen to text
    "boolean": ["text"],
}


class DatabaseManager:
    """
    PostgreSQL database operations manager.
    Handles connections, table creation, and data insertion.
    """
    
    def __init__(self):
        self.connection_params = {
            "host": settings.postgres_host,
            "port": settings.postgres_port,
            "database": settings.postgres_db,
            "user": settings.postgres_user,
            "password": settings.postgres_password
        }
        logger.info(f"DatabaseManager initialized for database: {settings.postgres_db}")
    
    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections.
        
        Yields:
            psycopg2 connection object
        """
        conn = None
        try:
            conn = psycopg2.connect(**self.connection_params)
            logger.debug("Database connection established")
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {str(e)}")
            raise
        finally:
            if conn:
                conn.close()
                logger.debug("Database connection closed")
    
    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database.
        
        Args:
            table_name: Name of the table
            
        Returns:
            True if table exists, False otherwise
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = %s
                    );
                    """,
                    (table_name.lower(),)
                )
                exists = cur.fetchone()[0]
                logger.info(f"Table '{table_name}' exists: {exists}")
                return exists
    
    def create_table(self, table_name: str, column_types: Dict[str, str]) -> bool:
        """
        Create a PostgreSQL table with the specified schema.
        
        Args:
            table_name: Name of the table to create
            column_types: Dictionary mapping column names to PostgreSQL types
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Creating table '{table_name}' with {len(column_types)} columns")
        
        try:
            # Build CREATE TABLE statement
            columns_def = []
            for col_name, col_type in column_types.items():
                # Sanitize column name
                safe_col_name = self._sanitize_identifier(col_name)
                columns_def.append(f"{safe_col_name} {col_type}")
            
            create_statement = f"""
                CREATE TABLE IF NOT EXISTS {self._sanitize_identifier(table_name)} (
                    {', '.join(columns_def)}
                );
            """
            
            logger.debug(f"CREATE TABLE statement: {create_statement}")
            
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(create_statement)
            
            logger.info(f"Table '{table_name}' created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating table '{table_name}': {str(e)}")
            return False
    
    _ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")

    def _normalize_date_columns(self, df: pd.DataFrame, column_types: Dict[str, str]) -> pd.DataFrame:
        """
        Parse ambiguous date strings (e.g. DD-MM-YY) into ISO 8601 (YYYY-MM-DD)
        so PostgreSQL accepts them regardless of its datestyle setting.
        """
        date_type_keywords = ('DATE', 'TIMESTAMP')
        for col, col_type in column_types.items():
            if col not in df.columns:
                continue
            if not any(kw in col_type.upper() for kw in date_type_keywords):
                continue
            try:
                non_null = df[col].dropna()
                first_valid = str(non_null.astype(str).iloc[0]) if len(non_null) > 0 else ""
                already_iso = bool(self._ISO_DATE_RE.match(str(first_valid)))
                parsed = pd.to_datetime(
                    df[col],
                    dayfirst=not already_iso,
                    errors='coerce',
                )
                non_null_ratio = parsed.notna().sum() / max(len(parsed), 1)
                if non_null_ratio >= 0.5:
                    df[col] = parsed.dt.strftime('%Y-%m-%d').where(parsed.notna(), other=None)
                    logger.info(f"Normalized date column '{col}' to ISO format ({non_null_ratio:.0%} parsed)")
                else:
                    logger.warning(f"Column '{col}' typed as {col_type} but only {non_null_ratio:.0%} parsed as dates, leaving as-is")
            except Exception as e:
                logger.warning(f"Could not normalize date column '{col}': {e}")
        return df

    def _widen_narrow_varchars(self, table_name: str, df: pd.DataFrame, column_types: Dict[str, str]) -> None:
        """
        Before insertion, check every VARCHAR(n) column: if the data contains values
        longer than n, ALTER the column to TEXT so the INSERT won't fail.
        """
        varchar_re = re.compile(r"VARCHAR\((\d+)\)", re.IGNORECASE)
        alterations: List[str] = []
        for col, col_type in column_types.items():
            m = varchar_re.search(col_type)
            if not m or col not in df.columns:
                continue
            limit = int(m.group(1))
            max_len = df[col].dropna().astype(str).str.len().max()
            if pd.isna(max_len):
                continue
            if int(max_len) > limit:
                safe_col = self._sanitize_identifier(col)
                alterations.append(safe_col)
                logger.warning(f"Column '{col}' max length {int(max_len)} exceeds VARCHAR({limit}); widening to TEXT")
        if alterations:
            safe_table = self._sanitize_identifier(table_name)
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    for safe_col in alterations:
                        cur.execute(f"ALTER TABLE {safe_table} ALTER COLUMN {safe_col} TYPE TEXT")

    @staticmethod
    def _is_numeric_overflow_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        code = getattr(exc, "pgcode", None)
        return code == "22003" and "numeric" in msg and "overflow" in msg

    @staticmethod
    def _is_integer_overflow_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        code = getattr(exc, "pgcode", None)
        return code == "22003" and (
            "integer out of range" in msg or "bigint out of range" in msg
        )

    def _widen_integer_columns_for_retry(
        self,
        table_name: str,
        df: pd.DataFrame,
        widen_bigint_to_numeric: bool = False,
    ) -> List[str]:
        """
        Widen integer-family columns for overflow recovery.

        - smallint/integer -> BIGINT
        - bigint -> NUMERIC (only when widen_bigint_to_numeric=True)
        """
        widened: List[str] = []
        df_cols = {str(col).lower() for col in df.columns}
        safe_table = self._sanitize_identifier(table_name)

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT column_name, data_type
                        FROM information_schema.columns
                        WHERE table_schema = 'public'
                          AND table_name = %s
                          AND data_type IN ('smallint', 'integer', 'bigint')
                        """,
                        (table_name.lower(),),
                    )
                    int_cols = cur.fetchall()

                    for col_name, data_type in int_cols:
                        if col_name.lower() not in df_cols:
                            continue

                        target_type = None
                        if data_type in ("smallint", "integer"):
                            target_type = "BIGINT"
                        elif data_type == "bigint" and widen_bigint_to_numeric:
                            target_type = "NUMERIC"

                        if not target_type:
                            continue

                        safe_col = self._sanitize_identifier(col_name)
                        cur.execute(
                            f"ALTER TABLE {safe_table} ALTER COLUMN {safe_col} TYPE {target_type} USING {safe_col}::{target_type}"
                        )
                        widened.append(col_name)

            if widened:
                logger.warning(
                    f"Widened integer columns for retry in '{table_name}': {widened}"
                )
            return widened
        except Exception as e:
            logger.error(f"Failed to widen integer columns for '{table_name}': {e}")
            return []

    def _widen_numeric_columns_for_retry(self, table_name: str, df: pd.DataFrame) -> List[str]:
        """
        Widen numeric/decimal columns present in the incoming DataFrame to
        unconstrained NUMERIC so inserts do not fail on precision overflow
        (e.g. NUMERIC(15,2) receiving larger values).
        """
        widened: List[str] = []
        df_cols = {_normal_col.lower(): _normal_col for _normal_col in df.columns}
        safe_table = self._sanitize_identifier(table_name)

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_schema = 'public'
                          AND table_name = %s
                          AND data_type IN ('numeric', 'decimal')
                        """,
                        (table_name.lower(),),
                    )
                    numeric_cols = [row[0] for row in cur.fetchall()]

                    for col in numeric_cols:
                        if col.lower() not in df_cols:
                            continue
                        safe_col = self._sanitize_identifier(col)
                        cur.execute(
                            f"ALTER TABLE {safe_table} ALTER COLUMN {safe_col} TYPE NUMERIC USING {safe_col}::NUMERIC"
                        )
                        widened.append(col)

            if widened:
                logger.warning(
                    f"Widened numeric columns to unconstrained NUMERIC for retry in '{table_name}': {widened}"
                )
            return widened
        except Exception as e:
            logger.error(f"Failed to widen numeric columns for '{table_name}': {e}")
            return []

    def insert_data(
        self,
        table_name: str,
        df: pd.DataFrame,
        batch_size: int = 1000,
        column_types: Optional[Dict[str, str]] = None
    ) -> int:
        """
        Insert DataFrame data into PostgreSQL table using batch operations.
        
        Args:
            table_name: Name of the target table
            df: DataFrame to insert
            batch_size: Number of rows per batch
            column_types: Optional dict of column→PostgreSQL type for date normalization
            
        Returns:
            Number of rows inserted
        """
        logger.info(f"Inserting {len(df)} rows into table '{table_name}'")
        
        try:
            if column_types:
                df = self._normalize_date_columns(df.copy(), column_types)
                self._widen_narrow_varchars(table_name, df, column_types)

            # Prepare column names and data
            columns = df.columns.tolist()
            safe_columns = [self._sanitize_identifier(col) for col in columns]
            
            # Convert DataFrame to list of tuples
            data = [tuple(row) for row in df.values]
            
            # Build INSERT statement
            placeholders = ', '.join(['%s'] * len(columns))
            insert_statement = f"""
                INSERT INTO {self._sanitize_identifier(table_name)} 
                ({', '.join(safe_columns)})
                VALUES ({placeholders})
            """
            
            logger.debug(f"INSERT statement template: {insert_statement}")
            
            overflow_retry_count = 0
            max_overflow_retries = 3
            while True:
                try:
                    with self.get_connection() as conn:
                        with conn.cursor() as cur:
                            # Use execute_batch for better performance
                            execute_batch(cur, insert_statement, data, page_size=batch_size)
                    break
                except Exception as e:
                    if overflow_retry_count < max_overflow_retries:
                        if self._is_integer_overflow_error(e):
                            widen_bigint = "bigint out of range" in str(e).lower()
                            widened = self._widen_integer_columns_for_retry(
                                table_name,
                                df,
                                widen_bigint_to_numeric=widen_bigint,
                            )
                            if widened:
                                overflow_retry_count += 1
                                logger.warning(
                                    f"Retrying insert into '{table_name}' after integer widening "
                                    f"(attempt {overflow_retry_count}/{max_overflow_retries})"
                                )
                                continue
                        if self._is_numeric_overflow_error(e):
                            widened = self._widen_numeric_columns_for_retry(table_name, df)
                            if widened:
                                overflow_retry_count += 1
                                logger.warning(
                                    f"Retrying insert into '{table_name}' after numeric widening "
                                    f"(attempt {overflow_retry_count}/{max_overflow_retries})"
                                )
                                continue
                    raise
            
            logger.info(f"Successfully inserted {len(df)} rows into '{table_name}'")
            return len(df)
            
        except Exception as e:
            logger.error(f"Error inserting data into '{table_name}': {str(e)}")
            raise
    
    def _sanitize_identifier(self, identifier: str) -> str:
        """
        Sanitize SQL identifier (table/column name) to prevent SQL injection.
        Converts to lowercase for PostgreSQL compatibility (no quotes needed).
        
        Args:
            identifier: Raw identifier
            
        Returns:
            Sanitized lowercase identifier wrapped in quotes
        """
        # Remove any existing quotes and convert to lowercase
        identifier = identifier.replace('"', '').lower()
        # Wrap in double quotes (PostgreSQL will store as lowercase)
        return f'"{identifier}"'
    
    def get_table_info(self, table_name: str) -> Optional[Dict]:
        """
        Get information about a table (columns, row count).
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dictionary with table information or None if table doesn't exist
        """
        if not self.table_exists(table_name):
            return None
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Get column information
                    cur.execute(
                        """
                        SELECT column_name, data_type 
                        FROM information_schema.columns 
                        WHERE table_schema = 'public' 
                        AND table_name = %s
                        ORDER BY ordinal_position;
                        """,
                        (table_name.lower(),)
                    )
                    columns = cur.fetchall()
                    
                    # Get row count
                    cur.execute(
                        sql.SQL("SELECT COUNT(*) FROM {}").format(
                            sql.Identifier(table_name)
                        )
                    )
                    row_count = cur.fetchone()[0]
                    
                    return {
                        "columns": [{"name": col[0], "type": col[1]} for col in columns],
                        "row_count": row_count
                    }
                    
        except Exception as e:
            logger.error(f"Error getting table info for '{table_name}': {str(e)}")
            return None
    
    @staticmethod
    def _columns_to_str(columns) -> str:
        """Normalize columns to a comma-separated string. Handles list of dicts (e.g. [{'name': 'x'}])."""
        if columns is None:
            return ""
        if isinstance(columns, str):
            return columns
        if isinstance(columns, (list, tuple)):
            parts = []
            for item in columns:
                if isinstance(item, dict):
                    parts.append(str(item.get("name", item.get("column_name", str(item)))))
                else:
                    parts.append(str(item))
            return ", ".join(parts)
        return str(columns)

    def insert_tables_metadata(self, metadata: Dict) -> bool:
        """Insert metadata record into tables_metadata table."""
        logger.info(f"Inserting metadata for table '{metadata.get('table_name')}'")
        
        try:
            insert_statement = """
                INSERT INTO tables_metadata (
                    data_domain, table_name, columns, comments, source, source_url,
                    released_on, updated_on, rows_count, business_metadata, table_view,
                    period_cols, min_period_sql, max_period_sql
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            values = (
                metadata.get('data_domain'), metadata.get('table_name'),
                self._columns_to_str(metadata.get('columns')),
                metadata.get('comments'), metadata.get('source'), metadata.get('source_url'),
                metadata.get('released_on'), metadata.get('updated_on'), metadata.get('rows_count'),
                metadata.get('business_metadata'), metadata.get('table_view', 'table'),
                metadata.get('period_cols'), metadata.get('min_period_sql'), metadata.get('max_period_sql')
            )
            
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(insert_statement, values)
            
            logger.info(f"Successfully inserted metadata for '{metadata.get('table_name')}'")
            return True
        except Exception as e:
            logger.error(f"Error inserting tables_metadata: {str(e)}")
            return False
    
    def insert_operational_metadata(self, metadata: Dict) -> bool:
        """Insert/update metadata record into operational_metadata table."""
        logger.info(f"Inserting operational metadata for table '{metadata.get('table_name')}'")
        
        try:
            insert_statement = """
                INSERT INTO operational_metadata (
                    table_name, table_view, period_cols, first_available_value,
                    last_available_value, last_updated_on, rows_count, columns,
                    source_url, business_metadata, major_domain, sub_domain, brief_summary
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (table_name)
                DO UPDATE SET
                    table_view = COALESCE(EXCLUDED.table_view, operational_metadata.table_view),
                    period_cols = COALESCE(EXCLUDED.period_cols, operational_metadata.period_cols),
                    first_available_value = COALESCE(EXCLUDED.first_available_value, operational_metadata.first_available_value),
                    last_available_value = COALESCE(EXCLUDED.last_available_value, operational_metadata.last_available_value),
                    last_updated_on = COALESCE(EXCLUDED.last_updated_on, operational_metadata.last_updated_on),
                    rows_count = COALESCE(EXCLUDED.rows_count, operational_metadata.rows_count),
                    columns = COALESCE(EXCLUDED.columns, operational_metadata.columns),
                    source_url = COALESCE(EXCLUDED.source_url, operational_metadata.source_url),
                    business_metadata = COALESCE(EXCLUDED.business_metadata, operational_metadata.business_metadata),
                    major_domain = COALESCE(EXCLUDED.major_domain, operational_metadata.major_domain),
                    sub_domain = COALESCE(EXCLUDED.sub_domain, operational_metadata.sub_domain),
                    brief_summary = COALESCE(EXCLUDED.brief_summary, operational_metadata.brief_summary)
            """
            
            values = (
                metadata.get('table_name'), metadata.get('table_view', 'Table'),
                metadata.get('period_cols'), metadata.get('first_available_value'),
                metadata.get('last_available_value'), metadata.get('last_updated_on'),
                metadata.get('rows_count'), self._columns_to_str(metadata.get('columns')),
                metadata.get('source_url'), metadata.get('business_metadata'), metadata.get('major_domain'),
                metadata.get('sub_domain'), metadata.get('brief_summary')
            )
            
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(insert_statement, values)
            
            logger.info(f"Successfully inserted operational metadata for '{metadata.get('table_name')}'")
            return True
        except Exception as e:
            logger.error(f"Error inserting operational_metadata: {str(e)}")
            return False

    def list_tables_from_metadata(self) -> List[str]:
        """Return table_name from every row in tables_metadata (ingestion-created tables)."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT table_name FROM tables_metadata")
                    return [row[0] for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"Error listing tables from metadata: {e}")
            return []

    # ------------------------------------------------------------------
    # Schema evolution helpers
    # ------------------------------------------------------------------

    def get_primary_keys(self, table_name: str) -> List[str]:
        """
        Return the list of primary-key column names for an existing table.
        Queries the real PostgreSQL constraint catalogue so the result is always
        authoritative, regardless of how the table was created.

        Returns an empty list if no PK is defined (caller should fall back
        to append-only behaviour and log a warning).
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT kcu.column_name
                        FROM information_schema.table_constraints AS tc
                        JOIN information_schema.key_column_usage AS kcu
                          ON tc.constraint_name = kcu.constraint_name
                         AND tc.table_schema   = kcu.table_schema
                        WHERE tc.constraint_type = 'PRIMARY KEY'
                          AND tc.table_schema    = 'public'
                          AND tc.table_name      = %s
                        ORDER BY kcu.ordinal_position
                        """,
                        (table_name.lower(),)
                    )
                    pks = [row[0] for row in cur.fetchall()]
                    logger.info(f"Primary keys for '{table_name}': {pks}")
                    return pks
        except Exception as e:
            logger.error(f"Error fetching primary keys for '{table_name}': {e}")
            return []

    def get_table_column_types(self, table_name: str) -> Dict[str, str]:
        """
        Return a mapping of {column_name_lowercase: postgres_data_type} for
        every column in the table.  Uses information_schema for portability.
        Returns an empty dict if the table does not exist or on error.
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT column_name, data_type
                        FROM information_schema.columns
                        WHERE table_schema = 'public'
                          AND table_name   = %s
                        ORDER BY ordinal_position
                        """,
                        (table_name.lower(),)
                    )
                    result = {row[0].lower(): row[1].lower() for row in cur.fetchall()}
                    logger.info(f"Fetched {len(result)} column types for '{table_name}'")
                    return result
        except Exception as e:
            logger.error(f"Error fetching column types for '{table_name}': {e}")
            return {}

    def alter_table_add_columns(
        self,
        table_name: str,
        new_columns: Dict[str, str]  # {col_name: postgres_type}
    ) -> List[str]:
        """
        Add missing columns to an existing table.  New columns are nullable by
        default so that existing rows receive NULL for the new field.

        Returns the list of column names that were successfully added.
        """
        added: List[str] = []
        if not new_columns:
            return added
        safe_table = self._sanitize_identifier(table_name)
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    for col_name, col_type in new_columns.items():
                        safe_col = self._sanitize_identifier(col_name)
                        try:
                            cur.execute(
                                f"ALTER TABLE {safe_table} ADD COLUMN IF NOT EXISTS {safe_col} {col_type}"
                            )
                            added.append(col_name)
                            logger.info(
                                f"ALTER TABLE {table_name}: added column '{col_name}' ({col_type})"
                            )
                        except Exception as col_err:
                            logger.warning(
                                f"Could not add column '{col_name}' to '{table_name}': {col_err}"
                            )
        except Exception as e:
            logger.error(f"Error altering table '{table_name}': {e}")
        return added

    def alter_column_type_safe(
        self,
        table_name: str,
        col_name: str,
        new_type: str
    ) -> bool:
        """
        Widen a column's type using ALTER COLUMN … TYPE … USING.
        Only safe widenings are attempted; the caller is responsible for
        checking _WIDENING_GROUPS before calling this.

        Returns True on success, False on failure.
        """
        safe_table = self._sanitize_identifier(table_name)
        safe_col   = self._sanitize_identifier(col_name)
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"ALTER TABLE {safe_table} "
                        f"ALTER COLUMN {safe_col} TYPE {new_type} "
                        f"USING {safe_col}::{new_type}"
                    )
            logger.info(f"Widened '{table_name}'.'{col_name}' to {new_type}")
            return True
        except Exception as e:
            logger.warning(
                f"Could not widen '{table_name}'.'{col_name}' to {new_type}: {e}"
            )
            return False

    def ensure_ingested_at_column(self, table_name: str) -> None:
        """
        Add `ingested_at TIMESTAMPTZ` to the table if it doesn't already exist.
        This column is maintained by the ingestion pipeline as an audit trail.
        """
        existing = self.get_table_column_types(table_name)
        if "ingested_at" not in existing:
            self.alter_table_add_columns(
                table_name,
                {"ingested_at": "TIMESTAMP WITH TIME ZONE"}
            )
            logger.info(f"Added 'ingested_at' audit column to '{table_name}'")

    def upsert_data(
        self,
        table_name: str,
        df: pd.DataFrame,
        key_columns: List[str],
        column_types: Optional[Dict[str, str]] = None,
        batch_size: int = 1000
    ) -> Tuple[int, int]:
        """
        Perform a deterministic UPSERT using INSERT … ON CONFLICT DO UPDATE.

        * On conflict (matching key_columns), all non-key columns are updated.
        * The `ingested_at` column is always refreshed to NOW().
        * Running the same data twice is fully idempotent.

        Args:
            table_name:   Target table (must exist).
            df:           Aligned DataFrame (columns already match the table).
            key_columns:  Columns that form the conflict target (PK / unique).
            column_types: Optional type hints for date normalization.
            batch_size:   Row batch size for execute_values.

        Returns:
            Tuple (rows_inserted, rows_updated) — estimated via xmax heuristic.
        """
        if df.empty:
            return 0, 0

        if column_types:
            df = self._normalize_date_columns(df.copy(), column_types)
            self._widen_narrow_varchars(table_name, df, column_types)

        # Stamp ingested_at for every row
        df = df.copy()
        df["ingested_at"] = datetime.utcnow()

        # Identify columns
        all_cols = [c for c in df.columns]
        key_cols_lower = [k.lower() for k in key_columns]
        update_cols = [c for c in all_cols if c.lower() not in key_cols_lower]

        if not update_cols:
            logger.warning(
                f"All columns are key columns for '{table_name}'; "
                "cannot build ON CONFLICT DO UPDATE. Falling back to INSERT IGNORE."
            )
            # Just do INSERT … ON CONFLICT DO NOTHING for identity tables
            return self._insert_on_conflict_nothing(table_name, df, key_cols_lower, batch_size)

        safe_table   = self._sanitize_identifier(table_name)
        safe_all     = [self._sanitize_identifier(c) for c in all_cols]
        safe_keys    = [self._sanitize_identifier(k) for k in key_cols_lower]
        safe_updates = [self._sanitize_identifier(c) for c in update_cols]

        # Build the conflict target — for PostgreSQL we need quoted identifiers
        conflict_target = ", ".join(safe_keys)
        update_set = ", ".join(f"{sc} = EXCLUDED.{sc}" for sc in safe_updates)

        insert_sql = (
            f"INSERT INTO {safe_table} ({', '.join(safe_all)}) "
            f"VALUES %s "
            f"ON CONFLICT ({conflict_target}) DO UPDATE SET {update_set} "
            f"RETURNING (xmax = 0) AS inserted"
        )

        rows_data = [tuple(row) for row in df.itertuples(index=False, name=None)]

        inserted_count = 0
        updated_count  = 0

        try:
            overflow_retry_count = 0
            max_overflow_retries = 3
            while True:
                try:
                    inserted_count = 0
                    updated_count = 0
                    with self.get_connection() as conn:
                        with conn.cursor() as cur:
                            for i in range(0, len(rows_data), batch_size):
                                batch = rows_data[i : i + batch_size]
                                execute_values(cur, insert_sql, batch)
                                results = cur.fetchall()
                                for (was_inserted,) in results:
                                    if was_inserted:
                                        inserted_count += 1
                                    else:
                                        updated_count += 1
                    break
                except Exception as e:
                    if overflow_retry_count < max_overflow_retries:
                        if self._is_integer_overflow_error(e):
                            widen_bigint = "bigint out of range" in str(e).lower()
                            widened = self._widen_integer_columns_for_retry(
                                table_name,
                                df,
                                widen_bigint_to_numeric=widen_bigint,
                            )
                            if widened:
                                overflow_retry_count += 1
                                logger.warning(
                                    f"Retrying upsert into '{table_name}' after integer widening "
                                    f"(attempt {overflow_retry_count}/{max_overflow_retries})"
                                )
                                continue
                        if self._is_numeric_overflow_error(e):
                            widened = self._widen_numeric_columns_for_retry(table_name, df)
                            if widened:
                                overflow_retry_count += 1
                                logger.warning(
                                    f"Retrying upsert into '{table_name}' after numeric widening "
                                    f"(attempt {overflow_retry_count}/{max_overflow_retries})"
                                )
                                continue
                    raise

            logger.info(
                f"Upsert into '{table_name}': {inserted_count} inserted, "
                f"{updated_count} updated"
            )
            return inserted_count, updated_count

        except Exception as e:
            logger.error(f"Error during upsert into '{table_name}': {e}")
            raise

    def _insert_on_conflict_nothing(
        self,
        table_name: str,
        df: pd.DataFrame,
        key_cols_lower: List[str],
        batch_size: int
    ) -> Tuple[int, int]:
        """Fallback: INSERT … ON CONFLICT DO NOTHING (for identity-only tables)."""
        safe_table = self._sanitize_identifier(table_name)
        all_cols   = df.columns.tolist()
        safe_all   = [self._sanitize_identifier(c) for c in all_cols]
        safe_keys  = [self._sanitize_identifier(k) for k in key_cols_lower]
        conflict_target = ", ".join(safe_keys)

        insert_sql = (
            f"INSERT INTO {safe_table} ({', '.join(safe_all)}) "
            f"VALUES %s ON CONFLICT ({conflict_target}) DO NOTHING"
        )
        rows_data = [tuple(row) for row in df.itertuples(index=False, name=None)]
        inserted = 0
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    for i in range(0, len(rows_data), batch_size):
                        batch = rows_data[i : i + batch_size]
                        execute_values(cur, insert_sql, batch)
                        inserted += cur.rowcount
        except Exception as e:
            logger.error(f"Error in insert-on-conflict-nothing for '{table_name}': {e}")
            raise
        return inserted, 0

    def drop_ingestion_tables_and_metadata(self) -> Dict[str, int]:
        """
        Drop all tables listed in tables_metadata, then delete from tables_metadata
        and operational_metadata. Returns {"tables_dropped": n, "metadata_cleared": 1}.
        """
        dropped = 0
        try:
            tables = self.list_tables_from_metadata()
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    for table_name in tables:
                        try:
                            cur.execute(
                                sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                                    sql.Identifier(table_name)
                                )
                            )
                            dropped += 1
                            logger.info(f"Dropped table: {table_name}")
                        except Exception as e:
                            logger.warning(f"Could not drop {table_name}: {e}")
                    for meta_table in ("tables_metadata", "operational_metadata"):
                        try:
                            cur.execute(sql.SQL("DELETE FROM {}").format(sql.Identifier(meta_table)))
                        except Exception as e:
                            logger.debug(f"Could not clear {meta_table} (may not exist): {e}")
            return {"tables_dropped": dropped, "metadata_cleared": 1}
        except Exception as e:
            logger.error(f"Error in drop_ingestion_tables_and_metadata: {e}")
            raise

    def delete_table_and_metadata(self, table_name: str) -> bool:
        """
        Drop a specific table from the database and remove its metadata.
        
        Args:
            table_name: Name of the table
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Drop the table
                    cur.execute(
                        sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                            sql.Identifier(table_name)
                        )
                    )
                    logger.info(f"Dropped table: {table_name}")
                    
                    # Clear from tables_metadata
                    cur.execute(
                        "DELETE FROM tables_metadata WHERE table_name = %s",
                        (table_name,)
                    )
                    
                    # Clear from operational_metadata
                    cur.execute(
                        "DELETE FROM operational_metadata WHERE table_name = %s",
                        (table_name,)
                    )
                    logger.info(f"Cleared metadata for table: {table_name}")
            return True
        except Exception as e:
            logger.error(f"Error in delete_table_and_metadata for {table_name}: {e}")
            return False


# Global database manager instance
db_manager = DatabaseManager()
