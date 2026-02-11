import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_batch
import pandas as pd
from typing import Dict, Optional, List
from contextlib import contextmanager
from app.config import settings
from app.core.logger import logger


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
    
    def insert_data(
        self,
        table_name: str,
        df: pd.DataFrame,
        batch_size: int = 1000
    ) -> int:
        """
        Insert DataFrame data into PostgreSQL table using batch operations.
        
        Args:
            table_name: Name of the target table
            df: DataFrame to insert
            batch_size: Number of rows per batch
            
        Returns:
            Number of rows inserted
        """
        logger.info(f"Inserting {len(df)} rows into table '{table_name}'")
        
        try:
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
            
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Use execute_batch for better performance
                    execute_batch(cur, insert_statement, data, page_size=batch_size)
            
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
                metadata.get('data_domain'), metadata.get('table_name'), metadata.get('columns'),
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
        """Insert metadata record into operational_metadata table."""
        logger.info(f"Inserting operational metadata for table '{metadata.get('table_name')}'")
        
        try:
            insert_statement = """
                INSERT INTO operational_metadata (
                    table_name, table_view, period_cols, first_available_value,
                    last_available_value, last_updated_on, rows_count, columns,
                    source_url, business_metadata, major_domain, sub_domain, brief_summary
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            values = (
                metadata.get('table_name'), metadata.get('table_view', 'Table'),
                metadata.get('period_cols'), metadata.get('first_available_value'),
                metadata.get('last_available_value'), metadata.get('last_updated_on'),
                metadata.get('rows_count'), metadata.get('columns'), metadata.get('source_url'),
                metadata.get('business_metadata'), metadata.get('major_domain'),
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


# Global database manager instance
db_manager = DatabaseManager()
