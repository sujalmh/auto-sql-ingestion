from typing import Dict, List, Optional, Tuple
from app.core.database import db_manager
from app.core.logger import logger
from app.core.column_utils import normalize_column_for_similarity


class SchemaValidator:
    """
    Validates schema compatibility between new files and existing tables.
    
    Compares column names from new files against existing table metadata
    to determine if incremental load is possible or if human approval is needed.
    """
    
    def __init__(self):
        logger.info("SchemaValidator initialized")
    
    def fetch_table_metadata(self, table_name: str) -> Optional[Dict]:
        """
        Fetch table metadata from tables_metadata table.
        
        Args:
            table_name: Name of the table to fetch metadata for
            
        Returns:
            Dictionary with table metadata or None if not found
        """
        logger.info(f"Fetching metadata for table: {table_name}")
        
        try:
            with db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    # Query tables_metadata for the table
                    cur.execute(
                        """
                        SELECT table_name, columns, data_domain, comments, rows_count
                        FROM tables_metadata
                        WHERE table_name = %s
                        """,
                        (table_name.lower(),)
                    )
                    
                    result = cur.fetchone()
                    
                    if not result:
                        logger.warning(f"No metadata found for table: {table_name}")
                        return None
                    
                    # Parse the columns field (comma-separated string)
                    columns_str = result[1]
                    columns_list = [col.strip() for col in columns_str.split(',') if col.strip()]
                    
                    metadata = {
                        'table_name': result[0],
                        'columns': columns_list,
                        'data_domain': result[2],
                        'comments': result[3],
                        'rows_count': result[4]
                    }
                    
                    logger.info(f"Metadata fetched: {len(columns_list)} columns, {result[4]} rows")
                    return metadata
                    
        except Exception as e:
            logger.error(f"Error fetching table metadata: {str(e)}")
            return None

    def find_similar_table_by_columns(
        self,
        new_columns: List[str],
        min_overlap: float = 0.7,
    ) -> Optional[Tuple[str, float]]:
        """
        Fallback when Milvus is unavailable or returns no results: find an existing
        table whose columns overlap with new_columns (after normalizing period/date
        in names) so that e.g. Jan upload can match Dec table without Milvus.
        Returns (table_name, score) or None.
        """
        tables = db_manager.list_tables_from_metadata()
        if not tables:
            logger.info("No tables in tables_metadata for fallback match")
            return None
        new_norm = {normalize_column_for_similarity(c) for c in new_columns}
        if not new_norm:
            return None
        best_table, best_score = None, 0.0
        for table_name in tables:
            meta = self.fetch_table_metadata(table_name)
            if not meta or not meta.get("columns"):
                continue
            existing_norm = {normalize_column_for_similarity(c) for c in meta["columns"]}
            if not existing_norm:
                continue
            overlap = len(new_norm & existing_norm) / len(existing_norm)
            if overlap >= min_overlap and overlap > best_score:
                best_score = overlap
                best_table = table_name
        if best_table is not None:
            logger.info(f"Fallback match: {best_table} (normalized column overlap: {best_score:.2%})")
        return (best_table, best_score) if best_table else None
    
    def validate_schema_match(
        self,
        new_columns: List[str],
        existing_columns: List[str]
    ) -> Dict:
        """
        Compare new file columns with existing table columns.
        
        Args:
            new_columns: List of column names from new file
            existing_columns: List of column names from existing table
            
        Returns:
            Dictionary with validation results:
            {
                'is_compatible': bool,
                'missing_columns': [],  # Columns in existing but not in new
                'extra_columns': [],    # Columns in new but not in existing
                'matching_columns': [], # Columns present in both
                'match_percentage': float
            }
        """
        logger.info(f"Validating schema match: {len(new_columns)} new vs {len(existing_columns)} existing")
        
        # Convert to sets for comparison
        new_set = set(col.lower() for col in new_columns)
        existing_set = set(col.lower() for col in existing_columns)
        
        # Find differences
        missing_columns = list(existing_set - new_set)  # In existing but not in new
        extra_columns = list(new_set - existing_set)    # In new but not in existing
        matching_columns = list(new_set & existing_set) # In both
        
        # Calculate match percentage
        if len(existing_set) > 0:
            match_percentage = (len(matching_columns) / len(existing_set)) * 100
        else:
            match_percentage = 0.0
        
        # Determine compatibility
        # Exact match required for auto-execution
        is_compatible = (len(missing_columns) == 0 and len(extra_columns) == 0)
        
        validation_result = {
            'is_compatible': is_compatible,
            'missing_columns': sorted(missing_columns),
            'extra_columns': sorted(extra_columns),
            'matching_columns': sorted(matching_columns),
            'match_percentage': round(match_percentage, 2)
        }
        
        logger.info(f"Validation result: compatible={is_compatible}, match={match_percentage:.2f}%")
        
        if not is_compatible:
            logger.warning(f"Schema mismatch detected:")
            if missing_columns:
                logger.warning(f"  - Missing columns: {missing_columns}")
            if extra_columns:
                logger.warning(f"  - Extra columns: {extra_columns}")
        
        return validation_result
    
    def generate_discrepancy_report(
        self,
        validation_result: Dict,
        table_name: str,
        new_file_name: str = "uploaded file"
    ) -> str:
        """
        Generate a human-readable discrepancy report.
        
        Args:
            validation_result: Result from validate_schema_match()
            table_name: Name of the existing table
            new_file_name: Name of the new file (optional)
            
        Returns:
            Formatted report string
        """
        logger.info(f"Generating discrepancy report for table: {table_name}")
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("SCHEMA VALIDATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Existing Table: {table_name}")
        report_lines.append(f"New File: {new_file_name}")
        report_lines.append(f"Match Percentage: {validation_result['match_percentage']}%")
        report_lines.append("")
        
        if validation_result['is_compatible']:
            report_lines.append("✓ COMPATIBLE - Schemas match exactly")
            report_lines.append(f"  All {len(validation_result['matching_columns'])} columns match")
        else:
            report_lines.append("✗ INCOMPATIBLE - Schema differences detected")
            report_lines.append("")
            
            if validation_result['missing_columns']:
                report_lines.append(f"Missing Columns ({len(validation_result['missing_columns'])}):")
                report_lines.append("  These columns exist in the table but not in the new file:")
                for col in validation_result['missing_columns']:
                    report_lines.append(f"    - {col}")
                report_lines.append("")
            
            if validation_result['extra_columns']:
                report_lines.append(f"Extra Columns ({len(validation_result['extra_columns'])}):")
                report_lines.append("  These columns exist in the new file but not in the table:")
                for col in validation_result['extra_columns']:
                    report_lines.append(f"    + {col}")
                report_lines.append("")
            
            if validation_result['matching_columns']:
                report_lines.append(f"Matching Columns ({len(validation_result['matching_columns'])}):")
                for col in validation_result['matching_columns'][:10]:  # Show first 10
                    report_lines.append(f"    ✓ {col}")
                if len(validation_result['matching_columns']) > 10:
                    report_lines.append(f"    ... and {len(validation_result['matching_columns']) - 10} more")
        
        report_lines.append("=" * 80)
        
        report = "\n".join(report_lines)
        logger.info("Discrepancy report generated")
        
        return report
    
    def validate_incremental_load(
        self,
        table_name: str,
        new_columns: List[str],
        new_file_name: str = "uploaded file"
    ) -> Tuple[bool, Dict, str]:
        """
        Complete validation workflow for incremental load.
        
        Args:
            table_name: Name of the existing table
            new_columns: List of column names from new file
            new_file_name: Name of the new file (optional)
            
        Returns:
            Tuple of (is_compatible, validation_result, report)
        """
        logger.info(f"Starting incremental load validation for: {table_name}")
        
        # Fetch existing table metadata
        metadata = self.fetch_table_metadata(table_name)
        
        if not metadata:
            logger.error(f"Cannot validate - table metadata not found: {table_name}")
            return False, {}, f"Error: Table '{table_name}' not found in metadata"
        
        # Validate schema match
        validation_result = self.validate_schema_match(new_columns, metadata['columns'])
        
        # Generate report
        report = self.generate_discrepancy_report(validation_result, table_name, new_file_name)
        
        return validation_result['is_compatible'], validation_result, report
    
    def compare_period_values(self, value1: str, value2: str) -> Tuple[bool, str]:
        """
        Compare two period values using hybrid approach.
        
        Tries in order:
        1. Date parsing (handles "2024-June", "Jun-2024", "2024-06", etc.)
        2. Numeric comparison (for years like "2024", "2023")
        3. String comparison (lexicographic fallback)
        
        Args:
            value1: First period value
            value2: Second period value
            
        Returns:
            Tuple of (value1 > value2, comparison_method_used)
        """
        from dateutil import parser as date_parser
        
        # Try date parsing first
        try:
            date1 = date_parser.parse(str(value1))
            date2 = date_parser.parse(str(value2))
            return date1 > date2, "date"
        except:
            pass
        
        # Try numeric comparison
        try:
            num1 = float(value1)
            num2 = float(value2)
            return num1 > num2, "numeric"
        except:
            pass
        
        # Fallback to string comparison
        return str(value1) > str(value2), "string"
    
    def detect_duplicate_data(
        self,
        table_name: str,
        new_df,
        period_column: Optional[str] = None
    ) -> Dict:
        """
        Detect if new data is duplicate by comparing period values.
        
        Args:
            table_name: Name of the existing table
            new_df: DataFrame with new data
            period_column: Name of the period column (if None, fetches from metadata)
            
        Returns:
            Dictionary with detection results:
            {
                'status': 'NEW_DATA' | 'DUPLICATE' | 'PARTIAL_OVERLAP' | 'NO_PERIOD_COLUMN',
                'existing_last_value': str,
                'new_first_value': str,
                'new_last_value': str,
                'comparison_method': str,
                'message': str
            }
        """
        logger.info(f"Detecting duplicate data for table: {table_name}")
        
        try:
            # Fetch operational metadata to get period column and last value
            with db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT period_cols, last_available_value
                        FROM operational_metadata
                        WHERE table_name = %s
                        """,
                        (table_name.lower(),)
                    )
                    
                    result = cur.fetchone()
                    
                    if not result:
                        logger.warning(f"No operational metadata found for table: {table_name}")
                        return {
                            'status': 'NO_PERIOD_COLUMN',
                            'message': 'No operational metadata found'
                        }
                    
                    existing_period_col = result[0]
                    existing_last_value = result[1]
                    
                    # Use provided period column or fall back to metadata
                    period_col = period_column or existing_period_col
                    
                    if not period_col or period_col not in new_df.columns:
                        logger.warning(f"Period column '{period_col}' not found in new data")
                        return {
                            'status': 'NO_PERIOD_COLUMN',
                            'message': f"Period column '{period_col}' not found"
                        }
                    
                    if not existing_last_value:
                        logger.info("No last_available_value in existing table, assuming new data")
                        return {
                            'status': 'NEW_DATA',
                            'message': 'No existing period data to compare'
                        }
                    
                    # Get unique period values from new data
                    new_period_values = new_df[period_col].dropna().unique()
                    
                    if len(new_period_values) == 0:
                        logger.warning("No period values found in new data")
                        return {
                            'status': 'NO_PERIOD_COLUMN',
                            'message': 'No period values in new data'
                        }
                    
                    # Sort period values
                    try:
                        # Try sorting as dates first
                        from dateutil import parser as date_parser
                        sorted_values = sorted(new_period_values, key=lambda x: date_parser.parse(str(x)))
                    except:
                        # Fallback to string sorting
                        sorted_values = sorted(new_period_values, key=str)
                    
                    new_first_value = str(sorted_values[0])
                    new_last_value = str(sorted_values[-1])
                    
                    logger.info(f"Existing last value: {existing_last_value}")
                    logger.info(f"New data range: {new_first_value} to {new_last_value}")
                    
                    # Compare periods
                    first_gt_last, method1 = self.compare_period_values(new_first_value, existing_last_value)
                    last_lte_last, method2 = self.compare_period_values(existing_last_value, new_last_value)
                    
                    if first_gt_last:
                        # New data starts after existing data ends
                        status = 'NEW_DATA'
                        message = f"New data ({new_first_value} to {new_last_value}) extends beyond existing data (up to {existing_last_value})"
                    elif last_lte_last:
                        # New data ends before or at existing data end
                        status = 'DUPLICATE'
                        message = f"Duplicate detected: New data ({new_first_value} to {new_last_value}) is already covered by existing data (up to {existing_last_value})"
                    else:
                        # Partial overlap
                        status = 'PARTIAL_OVERLAP'
                        message = f"Partial overlap: New data ({new_first_value} to {new_last_value}) overlaps with existing data (up to {existing_last_value})"
                    
                    result = {
                        'status': status,
                        'existing_last_value': existing_last_value,
                        'new_first_value': new_first_value,
                        'new_last_value': new_last_value,
                        'comparison_method': method1,
                        'message': message,
                        'period_column': period_col
                    }
                    
                    logger.info(f"Duplicate detection result: {status}")
                    return result
                    
        except Exception as e:
            logger.error(f"Error detecting duplicate data: {str(e)}")
            return {
                'status': 'NO_PERIOD_COLUMN',
                'message': f'Error: {str(e)}'
            }


# Global schema validator instance
schema_validator = SchemaValidator()
