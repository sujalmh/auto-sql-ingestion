import pandas as pd
import re
from datetime import date
from typing import Dict, List, Optional
from app.core.logger import logger


class DataPreprocessor:
    """
    Data preprocessing module for transforming Excel/CSV files.
    Handles header merging, date column transformation, and data cleaning.
    """
    
    def __init__(self):
        logger.info("DataPreprocessor initialized")
    
    def merge_headers(self, df: pd.DataFrame, strategy: Dict) -> pd.DataFrame:
        """
        Merge multi-level headers if needed.
        Handles both pandas MultiIndex columns and multi-level headers in rows.
        
        Args:
            df: Input DataFrame
            strategy: Strategy dictionary from LLM analysis
            
        Returns:
            DataFrame with merged headers
        """
        if not strategy.get("needs_header_merge", False):
            logger.info("No header merging needed")
            return df
        
        logger.info("Merging multi-level headers")
        
        # Case 1: DataFrame has MultiIndex columns (pandas detected it)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = self._merge_multiindex_columns(df.columns)
            logger.info(f"Merged MultiIndex columns: {df.columns.tolist()}")
            return df
        
        # Case 2: Multi-level headers stored in first rows (common in Excel)
        # Detect if first few rows contain header information
        df = self._detect_and_merge_row_headers(df)
        
        return df
    
    def _merge_multiindex_columns(self, columns: pd.MultiIndex) -> list:
        """
        Merge pandas MultiIndex columns into single-level names.
        
        Args:
            columns: MultiIndex columns
            
        Returns:
            List of merged column names
        """
        merged = []
        for col in columns:
            # Filter out empty/unnamed levels and join with underscore
            parts = [str(part).strip() for part in col if str(part).strip() and not str(part).startswith('Unnamed')]
            merged_name = '_'.join(parts) if parts else 'unnamed_column'
            merged.append(merged_name)
        
        logger.info(f"Merged {len(columns)} MultiIndex columns")
        return merged
    
    def _detect_and_merge_row_headers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and merge multi-level headers stored in the first rows.
        
        This handles cases where Excel files have headers like:
        Row 0: | Year | Unit value index | Unit value index | Quantum index | ...
        Row 1: |      | Exports          | Imports          | Exports       | ...
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with merged headers and header rows removed
        """
        # Check if first row(s) might be headers by looking for:
        # 1. Many duplicate values (indicating grouped headers)
        # 2. Many NaN/empty values
        # 3. Non-numeric values when rest of data is numeric
        
        max_header_rows = min(3, len(df))  # Check up to 3 rows
        potential_header_rows = []
        
        for i in range(max_header_rows):
            row = df.iloc[i]
            
            # Count unique non-null values
            non_null_values = row.dropna().astype(str)
            unique_ratio = len(non_null_values.unique()) / len(row) if len(row) > 0 else 1
            null_ratio = row.isna().sum() / len(row) if len(row) > 0 else 0
            
            # If row has many duplicates or many nulls, likely a header row
            if unique_ratio < 0.7 or null_ratio > 0.3:
                potential_header_rows.append(i)
                logger.debug(f"Row {i} identified as potential header (unique_ratio={unique_ratio:.2f}, null_ratio={null_ratio:.2f})")
        
        if not potential_header_rows:
            logger.info("No multi-level row headers detected")
            return df
        
        # Extract header rows
        header_rows = df.iloc[potential_header_rows]
        
        # Build new column names by merging header rows
        new_columns = []
        for col_idx in range(len(df.columns)):
            parts = []
            current_group = None
            
            for row_idx in potential_header_rows:
                value = str(header_rows.iloc[row_idx - potential_header_rows[0], col_idx])
                
                # Skip NaN, empty, or unnamed values
                if value and value.lower() not in ['nan', 'none', ''] and not value.startswith('Unnamed'):
                    # If this value is different from previous, it's a new level
                    if value != current_group:
                        parts.append(value.strip())
                        current_group = value
            
            # If no valid parts found, use original column name
            if not parts:
                parts = [str(df.columns[col_idx])]
            
            # Join parts with underscore
            merged_name = '_'.join(parts)
            new_columns.append(merged_name)
        
        # Apply new column names
        df.columns = new_columns
        
        # Remove the header rows from data
        df = df.iloc[max(potential_header_rows) + 1:].reset_index(drop=True)
        
        logger.info(f"Merged {len(potential_header_rows)} header rows into column names")
        logger.info(f"New columns: {df.columns.tolist()}")
        
        return df
    
    def transform_date_columns(
        self,
        df: pd.DataFrame,
        date_columns: List[str],
        analysis: Dict
    ) -> pd.DataFrame:
        """
        Transform date/timestamp columns from wide to long format.
        Converts columns like 'Jan-2023', 'Feb-2023' into rows with a date column.
        
        Args:
            df: Input DataFrame
            date_columns: List of column names containing dates
            analysis: Analysis results from LLM
            
        Returns:
            Transformed DataFrame
        """
        if not date_columns:
            logger.info("No date column transformation needed")
            return df
        
        logger.info(f"Transforming date columns: {date_columns}")
        
        # After merge_headers, MultiIndex columns are converted to strings
        # We need to find the actual column names in the DataFrame that match the date columns
        actual_date_columns = []
        for date_col in date_columns:
            # Check if the exact column exists
            if date_col in df.columns:
                actual_date_columns.append(date_col)
            else:
                # For merged MultiIndex, find columns containing date info
                date_col_str = str(date_col)
                matching_cols = [col for col in df.columns if date_col_str in str(col)]
                if matching_cols:
                    actual_date_columns.extend(matching_cols)
                else:
                    logger.warning(f"Date column '{date_col}' not found in DataFrame")
        
        if not actual_date_columns:
            logger.warning("No matching date columns found, skipping transformation")
            return df
        
        logger.info(f"Matched date columns: {actual_date_columns}")
        
        # Identify non-date columns (these will be ID/dimension columns)
        id_columns = [col for col in df.columns if col not in actual_date_columns]
        
        if not id_columns:
            logger.warning("No ID columns found, using index")
            df['id'] = df.index
            id_columns = ['id']
        
        # Melt the DataFrame to convert date columns to rows
        df_melted = pd.melt(
            df,
            id_vars=id_columns,
            value_vars=actual_date_columns,
            var_name='period',
            value_name='value'
        )
        
        # Try to parse the period column as dates
        df_melted['period'] = self._parse_period_column(df_melted['period'])
        
        logger.info(f"Transformed shape: {df.shape} → {df_melted.shape}")
        return df_melted
    
    def _parse_period_column(self, period_series: pd.Series) -> pd.Series:
        """
        Parse period column to standardized date format.
        Handles formats like: 'Jan-2023', 'Q1-2024', '2023', 'Jan 2023', etc.
        
        Args:
            period_series: Series containing period strings
            
        Returns:
            Parsed series (as datetime or original if parsing fails)
        """
        try:
            # Try pandas to_datetime with various formats
            parsed = pd.to_datetime(period_series, errors='coerce')
            
            # If most values parsed successfully, use it
            if parsed.notna().sum() / len(parsed) > 0.8:
                logger.info("Successfully parsed period column as datetime")
                return parsed
            
            # Otherwise, keep as string
            logger.info("Keeping period column as string")
            return period_series
            
        except Exception as e:
            logger.warning(f"Could not parse period column: {str(e)}")
            return period_series
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        General data cleaning operations.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning data")
        
        original_shape = df.shape
        
        # Drop completely empty columns (all NaN or empty strings)
        empty_columns = []
        for idx, col in enumerate(df.columns):
            # Check if all values are NaN or empty strings
            # Use iloc to avoid issues with duplicate column names
            col_data = df.iloc[:, idx]
            is_all_nan = col_data.isna().all()
            is_all_empty = (col_data.astype(str).str.strip() == '').all()
            
            if is_all_nan or is_all_empty:
                empty_columns.append(col)
        
        if empty_columns:
            df = df.drop(columns=empty_columns)
            logger.info(f"Dropped {len(empty_columns)} empty columns: {empty_columns}")
        
        # Strip whitespace from string columns
        for idx, col in enumerate(df.columns):
            if df.iloc[:, idx].dtype == 'object':
                df.iloc[:, idx] = df.iloc[:, idx].astype(str).str.strip()
        
        # Replace empty strings with None
        df = df.replace('', None)
        df = df.replace('nan', None)
        
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        # Clean column names: remove special characters, replace spaces with underscores
        df.columns = [self._clean_column_name(col) for col in df.columns]
        
        logger.info(f"Data cleaning completed: {original_shape} → {df.shape}")
        return df
    
    def _clean_column_name(self, name: str) -> str:
        """
        Clean column name for database compatibility.
        Removes parenthetical text, special annotations, and standardizes format.
        
        Args:
            name: Original column name
            
        Returns:
            Cleaned column name
        """
        # Convert to string
        name = str(name)
        
        # Extract and preserve important keywords from parentheses before removing them
        # Keywords like "Adjusted", "Unadjusted", etc. should be preserved
        important_keywords = ['adjusted', 'unadjusted', 'total', 'net', 'gross', 'actual', 
                             'estimated', 'provisional', 'revised', 'final', 'preliminary']
        preserved_keywords = []
        
        # Find all parenthetical content
        import re
        parenthetical_content = re.findall(r'\(([^)]*)\)', name)
        for content in parenthetical_content:
            content_lower = content.lower()
            for keyword in important_keywords:
                if keyword in content_lower:
                    preserved_keywords.append(keyword)
        
        # Remove parenthetical text (e.g., "Year (ending March)" -> "Year")
        name = re.sub(r'\([^)]*\)', '', name)
        
        # Remove square brackets and their contents
        name = re.sub(r'\[[^\]]*\]', '', name)
        
        # Remove common annotations/suffixes
        # e.g., "Year - ending March" -> "Year"
        name = re.sub(r'\s*[-–—]\s*.*$', '', name)
        
        # Convert to lowercase
        name = name.lower()
        
        # Add preserved keywords back
        if preserved_keywords:
            name = name + '_' + '_'.join(preserved_keywords)
        
        # Replace spaces and special characters with underscores
        name = re.sub(r'[^\w\s]', '_', name)
        name = re.sub(r'\s+', '_', name)
        name = re.sub(r'_+', '_', name)
        
        # Remove leading/trailing underscores
        name = name.strip('_')
        
        # Ensure it doesn't start with a number
        if name and name[0].isdigit():
            name = 'col_' + name
        
        return name or 'unnamed_column'
    
    def preprocess(
        self,
        df: pd.DataFrame,
        analysis: Dict
    ) -> tuple[pd.DataFrame, str]:
        """
        Main preprocessing pipeline.
        
        Args:
            df: Input DataFrame
            analysis: Analysis results from LLM
            
        Returns:
            Tuple of (processed DataFrame, preprocessing summary)
        """
        logger.info("Starting preprocessing pipeline")
        
        summary_steps = []
        original_col_count = len(df.columns)
        
        # Step 1: Merge headers if needed
        df = self.merge_headers(df, analysis)
        if analysis.get("needs_header_merge"):
            summary_steps.append("Merged multi-level headers")
        
        # Step 2: Transform date columns if needed
        date_columns = analysis.get("date_columns", [])
        if date_columns:
            df = self.transform_date_columns(df, date_columns, analysis)
            summary_steps.append(f"Transformed {len(date_columns)} date columns to rows")
        
        # Step 3: Clean data (includes dropping empty columns)
        df = self.clean_data(df)
        
        # Step 4: Add standard columns (refresh_date, data_period when determinable)
        df["refresh_date"] = date.today().isoformat()
        data_period = analysis.get("data_period")
        if data_period:
            df["data_period"] = self._normalize_mmm_yyyy(data_period)
        
        # Check if any columns were dropped
        dropped_cols = original_col_count - len(df.columns)
        if dropped_cols > 0:
            summary_steps.append(f"Dropped {dropped_cols} empty columns")
        
        summary_steps.append("Applied data cleaning (whitespace, empty rows, column names)")
        
        summary = " | ".join(summary_steps) if summary_steps else "No preprocessing needed"
        logger.info(f"Preprocessing completed: {summary}")
        
        return df, summary

    def _normalize_mmm_yyyy(self, value: str) -> str:
        """Convert period string to Mmm-yyyy (e.g. Dec 2025)."""
        if not value or not str(value).strip():
            return value
        value = str(value).strip()
        month_abbr = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
        if re.match(r"^\d{1,2}-\d{4}$", value):
            m, y = value.split("-")
            return f"{month_abbr[int(m) - 1]} {y}"
        if re.match(r"^\d{4}-\d{1,2}$", value):
            y, m = value.split("-")
            return f"{month_abbr[int(m) - 1]} {y}"
        year_m = re.search(r"\b(20\d{2})\b", value)
        months_lower = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
                        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
        month_num = None
        for name, num in months_lower.items():
            if name in value.lower():
                month_num = num
                break
        if year_m and month_num is not None:
            return f"{month_abbr[month_num - 1]} {year_m.group(1)}"
        return value


# Global preprocessor instance
preprocessor = DataPreprocessor()
