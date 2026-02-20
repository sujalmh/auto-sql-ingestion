from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime
from app.core.database import db_manager
from app.core.logger import logger


class IncrementalLoader:
    """
    Handles incremental data loading (append-only) to existing tables.
    
    Responsibilities:
    - Append new rows to existing tables
    - Update operational_metadata (row counts, last_updated_on)
    - Maintain data integrity during incremental loads
    """
    
    def __init__(self):
        logger.info("IncrementalLoader initialized")
    
    def get_table_columns(self, table_name: str) -> Optional[List[str]]:
        """
        Get the column names from an existing table in the correct order.
        
        Args:
            table_name: Name of the table
            
        Returns:
            List of column names in table order, or None if error
        """
        try:
            with db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    # Query PostgreSQL information_schema to get column names in order
                    cur.execute(
                        """
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_name = %s
                        ORDER BY ordinal_position
                        """,
                        (table_name.lower(),)
                    )
                    
                    columns = [row[0] for row in cur.fetchall()]
                    
                    if columns:
                        logger.info(f"Fetched {len(columns)} columns from table: {table_name}")
                        return columns
                    else:
                        logger.warning(f"No columns found for table: {table_name}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error fetching table columns: {str(e)}")
            return None
    
    def append_data(
        self,
        table_name: str,
        df: pd.DataFrame,
        column_types: Dict[str, str]
    ) -> bool:
        """
        Append data to an existing table.
        
        IMPORTANT: This method aligns the new DataFrame columns with the existing
        table schema to handle column name mismatches (e.g., 'baseyear' vs 'base_year').
        
        Args:
            table_name: Name of the existing table
            df: DataFrame with new data to append
            column_types: Dictionary mapping column names to PostgreSQL types (not used, kept for compatibility)
            
        Returns:
            True if append successful, False otherwise
        """
        logger.info(f"Appending {len(df)} rows to table: {table_name}")
        
        try:
            # Get existing table's column schema
            existing_columns = self.get_table_columns(table_name)
            
            if not existing_columns:
                logger.error(f"Cannot append - failed to fetch table schema for: {table_name}")
                return False
            
            # Map new DataFrame columns to existing table columns (case-insensitive)
            df_columns_lower = {col.lower(): col for col in df.columns}
            existing_columns_lower = [col.lower() for col in existing_columns]
            
            # Create column mapping
            column_mapping = {}
            missing_in_new = []
            
            for existing_col in existing_columns:
                existing_col_lower = existing_col.lower()
                
                if existing_col_lower in df_columns_lower:
                    # Found matching column (case-insensitive)
                    original_col = df_columns_lower[existing_col_lower]
                    if original_col != existing_col:
                        column_mapping[original_col] = existing_col
                        logger.info(f"Mapping column: '{original_col}' → '{existing_col}'")
                else:
                    missing_in_new.append(existing_col)
            
            if missing_in_new:
                logger.warning(f"Columns in table but not in new data: {missing_in_new}")
                logger.warning("These columns will be filled with NULL values")
            
            # Rename columns to match existing table schema
            if column_mapping:
                df = df.rename(columns=column_mapping)
                logger.info(f"Renamed {len(column_mapping)} columns to match table schema")
            
            # Reorder and select only columns that exist in the table
            # Add missing columns with NULL values
            aligned_df = pd.DataFrame()
            for col in existing_columns:
                if col in df.columns:
                    aligned_df[col] = df[col]
                else:
                    aligned_df[col] = None
                    logger.debug(f"Added NULL column: {col}")
            
            logger.info(f"Aligned DataFrame: {len(aligned_df.columns)} columns match table schema")
            
            # Use the existing insert_data method from DatabaseManager
            # This handles batching and proper data type conversion
            # Returns number of rows inserted
            rows_inserted = db_manager.insert_data(
                table_name=table_name,
                df=aligned_df,
                column_types=column_types
            )
            
            if rows_inserted > 0:
                logger.info(f"Successfully appended {rows_inserted} rows to {table_name}")
                return True
            else:
                logger.error(f"Failed to append data to {table_name}")
                return False
            
        except Exception as e:
            logger.error(f"Error appending data to {table_name}: {str(e)}")
            return False
    
    def update_operational_metadata(
        self,
        table_name: str,
        rows_added: int,
        last_available_value: Optional[str] = None
    ) -> bool:
        """
        Update operational_metadata after incremental load.
        
        Updates:
        - row_count (increment by rows_added)
        - last_updated_on (current timestamp)
        - last_available_value (optional - for time series data)
        
        Args:
            table_name: Name of the table
            rows_added: Number of rows added in this incremental load
            last_available_value: Latest value for tracking (e.g., latest date)
            
        Returns:
            True if update successful, False otherwise
        """
        logger.info(f"Updating operational metadata for {table_name}")
        
        try:
            with db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    # Build update query
                    if last_available_value:
                        cur.execute(
                            """
                            UPDATE operational_metadata
                            SET rows_count = rows_count + %s,
                                last_updated_on = %s,
                                last_available_value = %s
                            WHERE table_name = %s
                            """,
                            (rows_added, datetime.now(), last_available_value, table_name.lower())
                        )
                    else:
                        cur.execute(
                            """
                            UPDATE operational_metadata
                            SET rows_count = rows_count + %s,
                                last_updated_on = %s
                            WHERE table_name = %s
                            """,
                            (rows_added, datetime.now(), table_name.lower())
                        )
                    
                    conn.commit()
                    
                    if cur.rowcount > 0:
                        logger.info(f"Operational metadata updated: +{rows_added} rows")
                        return True
                    else:
                        logger.warning(f"No metadata row found for table: {table_name}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error updating operational metadata: {str(e)}")
            return False
    
    def get_current_row_count(self, table_name: str) -> Optional[int]:
        """
        Get current row count from operational_metadata.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Current row count or None if not found
        """
        try:
            with db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT rows_count
                        FROM operational_metadata
                        WHERE table_name = %s
                        """,
                        (table_name.lower(),)
                    )
                    
                    result = cur.fetchone()
                    
                    if result:
                        return result[0]
                    else:
                        logger.warning(f"No row count found for table: {table_name}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error fetching row count: {str(e)}")
            return None
    
    def perform_incremental_load(
        self,
        table_name: str,
        df: pd.DataFrame,
        column_types: Dict[str, str],
        last_available_value: Optional[str] = None
    ) -> Dict:
        """
        Complete incremental load workflow.
        
        Steps:
        1. Get current row count (before)
        2. Append data to table
        3. Update operational_metadata
        4. Return summary
        
        Args:
            table_name: Name of the existing table
            df: DataFrame with new data
            column_types: Column type mappings
            last_available_value: Optional tracking value
            
        Returns:
            Dictionary with load summary
        """
        logger.info(f"Starting incremental load for {table_name}")
        
        try:
            # Get current row count
            rows_before = self.get_current_row_count(table_name)
            if rows_before is None:
                rows_before = 0
            
            rows_to_add = len(df)
            
            logger.info(f"Current rows: {rows_before}, Adding: {rows_to_add}")
            
            # Append data
            append_success = self.append_data(table_name, df, column_types)
            
            if not append_success:
                return {
                    'success': False,
                    'error': 'Failed to append data',
                    'rows_before': rows_before,
                    'rows_added': 0,
                    'rows_after': rows_before
                }
            
            # Update metadata
            metadata_success = self.update_operational_metadata(
                table_name,
                rows_to_add,
                last_available_value
            )
            
            if not metadata_success:
                logger.warning("Data appended but metadata update failed")
            
            # Get final row count
            rows_after = self.get_current_row_count(table_name) or (rows_before + rows_to_add)
            
            summary = {
                'success': True,
                'table_name': table_name,
                'rows_before': rows_before,
                'rows_added': rows_to_add,
                'rows_after': rows_after,
                'metadata_updated': metadata_success,
                'last_available_value': last_available_value,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Incremental load completed: {rows_before} → {rows_after} rows")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error during incremental load: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'rows_before': rows_before if 'rows_before' in locals() else 0,
                'rows_added': 0,
                'rows_after': rows_before if 'rows_before' in locals() else 0
            }


# Global incremental loader instance
incremental_loader = IncrementalLoader()
