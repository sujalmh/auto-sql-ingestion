import pandas as pd
import re
from datetime import date
from typing import Dict, List, Optional
from app.core.logger import logger

# Footer patterns for detecting footnote/notes rows at the bottom of data
_FOOTER_RE = re.compile(
    r"^\s*(notes?\s*:|source\s*:|"
    r"\*|@|#|†|‡|§|"
    r"\d+\.\s+[A-Z]|"
    r"p\s*[-–:]\s*provisional|"
    r"re\s*[-–:]\s*revised|"
    r"be\s*[-–:]\s*budget)",
    re.IGNORECASE,
)


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
        Drops 'Unnamed:*' level entries (artifacts of empty cells) and
        joins the remaining parts with underscores.
        """
        merged = []
        for col in columns:
            parts = [
                str(part).strip()
                for part in col
                if str(part).strip()
                and not str(part).startswith('Unnamed')
                and str(part).strip().lower() != 'nan'
            ]
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

        # Regex-based fallback: catch any remaining columns that look like date
        # periods but weren't matched due to LLM whitespace / punctuation drift.
        matched_set = set(actual_date_columns)
        _date_regexes = [
            self._DATE_RANGE_RE, self._FY_RE, self._MON_YEAR_RE,
            self._QUARTER_YEAR_RE, self._PLAIN_YEAR_RE,
        ]
        for col in df.columns:
            if col in matched_set:
                continue
            col_str = str(col).strip()
            if any(rx.match(col_str) for rx in _date_regexes):
                actual_date_columns.append(col)
                matched_set.add(col)
                logger.info(f"Regex fallback matched date column: '{col}'")

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
        
        # Split the 'period' column into structured year/month/fiscal_year columns
        df_melted = self._split_period_to_columns(df_melted)
        
        logger.info(f"Transformed shape: {df.shape} → {df_melted.shape}")
        return df_melted
    
    # ── Regex patterns for period splitting ──
    _FY_RE = re.compile(
        r'^\s*(?:FY\s*)?'
        r'(\d{4})'
        r'\s*[–\-/]\s*'
        r'(\d{2,4})'
        r'\s*$',
        re.IGNORECASE,
    )
    _MON_YEAR_RE = re.compile(
        r'^\s*'
        r'(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|'
        r'jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)'
        r'[\s.\-,/]+'
        r'(\d{4})'
        r'\s*$',
        re.IGNORECASE,
    )
    _QUARTER_YEAR_RE = re.compile(
        r'^\s*Q([1-4])'
        r'[\s\-/]+'
        r'(\d{4})'
        r'\s*$',
        re.IGNORECASE,
    )
    _PLAIN_YEAR_RE = re.compile(r'^\s*(\d{4})\s*$')
    # Date range: "April 01, 2018 - May 31, 2018" → captures both start and end
    _MONTH_NAMES_ALT = (
        r'jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|'
        r'jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?'
    )
    _DATE_RANGE_RE = re.compile(
        r'^\s*'
        r'(' + _MONTH_NAMES_ALT + r')'   # group 1: start month
        r'\s+(\d{1,2})'                   # group 2: start day
        r'\s*,?\s*(\d{4})'                # group 3: start year
        r'\s*[-–]\s*'                      # dash separator
        r'(' + _MONTH_NAMES_ALT + r')'   # group 4: end month
        r'\s+(\d{1,2})'                   # group 5: end day
        r'\s*,?\s*(\d{4})'                # group 6: end year
        r'\s*$',
        re.IGNORECASE,
    )
    # Year range: "2018-19", "2018-2019", "2018 - 2020"
    _YEAR_RANGE_RE = re.compile(
        r'^\s*(\d{4})\s*[-–/]\s*(\d{2,4})\s*$'
    )

    def _coalesce_duplicate_named_columns(self, df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """
        Collapse duplicate columns with the same cleaned name by taking the
        first non-empty value across duplicates for each row.
        """
        match_positions = [idx for idx, col in enumerate(df.columns) if str(col) == column_name]
        if len(match_positions) <= 1:
            return df

        duplicate_block = df.iloc[:, match_positions].copy()
        normalized_block = duplicate_block.mask(
            duplicate_block.apply(lambda col: col.astype(str).str.strip() == '')
        )
        merged_series = normalized_block.bfill(axis=1).iloc[:, 0]

        first_position = match_positions[0]
        keep_positions = [idx for idx in range(df.shape[1]) if idx not in match_positions]
        df = df.iloc[:, keep_positions].copy()
        df.insert(first_position, column_name, merged_series)

        logger.info(
            f"Coalesced {len(match_positions)} duplicate '{column_name}' columns into one"
        )
        return df

    def _split_period_to_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Split a 'period' column (created by melt) into structured columns:
          - 2020-21 / 2021-2022 → fiscal_year (str)
          - Jan-2024 / January 2024 → year (int) + month (int 1-12)
          - Q1-2024 → year (int) + quarter (int 1-4)
          - 2024    → year (int)
        Falls back to keeping 'period' as-is when pattern is unrecognised.
        """
        if 'period' not in df.columns:
            return df

        df = self._coalesce_duplicate_named_columns(df, 'period')

        samples = df['period'].dropna().astype(str).str.strip()
        if samples.empty:
            return df

        # Pick the dominant pattern from a sample of up to 20 unique values
        unique_vals = samples.drop_duplicates().head(20)

        fy_hits = sum(1 for v in unique_vals if self._FY_RE.match(v))
        mon_hits = sum(1 for v in unique_vals if self._MON_YEAR_RE.match(v))
        qtr_hits = sum(1 for v in unique_vals if self._QUARTER_YEAR_RE.match(v))
        yr_hits = sum(1 for v in unique_vals if self._PLAIN_YEAR_RE.match(v))
        range_hits = sum(1 for v in unique_vals if self._DATE_RANGE_RE.match(v))

        total = len(unique_vals)
        threshold = 0.5  # >50% of unique values must match a single pattern

        if fy_hits / total >= threshold:
            # "2020-21" / "2021-2022" — could be fiscal year OR year range
            # Fiscal year: end is start+1 (e.g. 2020-21, 2021-2022)
            # Year range: end is start+2.. (e.g. 2018-2020)
            def _parse_fy(val):
                m = self._FY_RE.match(str(val).strip())
                if not m:
                    return None, None
                start = int(m.group(1))
                end_raw = m.group(2)
                end = int(end_raw)
                if len(end_raw) == 2:
                    end = (start // 100) * 100 + end
                return start, end

            parsed_fy = [_parse_fy(v) for v in unique_vals]
            all_fiscal = all(
                e is not None and e == s + 1
                for s, e in parsed_fy if s is not None
            )

            if all_fiscal:
                # True fiscal year: "2020-21" → fiscal_year="2020-21"
                def _to_fy(val):
                    m = self._FY_RE.match(str(val).strip())
                    if not m:
                        return val
                    start = m.group(1)
                    end = m.group(2)
                    if len(end) == 4:
                        end = end[2:]
                    return f"{start}-{end}"
                df['fiscal_year'] = df['period'].astype(str).str.strip().apply(_to_fy)
                df = df.drop(columns=['period'])
                logger.info(f"Split period → fiscal_year ({fy_hits}/{total} matched)")
            else:
                # Year range: "2018-2020" → from_year + to_year
                def _extract_year_range(val):
                    m = self._FY_RE.match(str(val).strip())
                    if not m:
                        return pd.NA, pd.NA
                    start = int(m.group(1))
                    end_raw = m.group(2)
                    end = int(end_raw)
                    if len(end_raw) == 2:
                        end = (start // 100) * 100 + end
                    return start, end
                extracted = df['period'].apply(
                    lambda v: pd.Series(_extract_year_range(v), index=['from_year', 'to_year'])
                )
                df['from_year'] = pd.to_numeric(extracted['from_year'], errors='coerce').astype('Int64')
                df['to_year'] = pd.to_numeric(extracted['to_year'], errors='coerce').astype('Int64')
                df = df.drop(columns=['period'])
                logger.info(f"Split period → from_year + to_year ({fy_hits}/{total} matched)")

        elif mon_hits / total >= threshold:
            # Month-year: "Jan-2024" → year=2024, month=1
            def _extract_month_year(val):
                m = self._MON_YEAR_RE.match(str(val).strip())
                if not m:
                    return pd.NA, pd.NA
                month_str = m.group(1).lower()[:3]
                month_num = self._MONTH_MAP.get(month_str)
                year_num = int(m.group(2))
                return year_num, month_num
            extracted = df['period'].apply(lambda v: pd.Series(_extract_month_year(v), index=['year', 'month']))
            df['year'] = pd.to_numeric(extracted['year'], errors='coerce').astype('Int64')
            df['month'] = pd.to_numeric(extracted['month'], errors='coerce').astype('Int64')
            df = df.drop(columns=['period'])
            logger.info(f"Split period → year + month ({mon_hits}/{total} matched)")

        elif qtr_hits / total >= threshold:
            # Quarter: "Q1-2024" → year=2024, quarter=1
            def _extract_qtr_year(val):
                m = self._QUARTER_YEAR_RE.match(str(val).strip())
                if not m:
                    return pd.NA, pd.NA
                return int(m.group(2)), int(m.group(1))
            extracted = df['period'].apply(lambda v: pd.Series(_extract_qtr_year(v), index=['year', 'quarter']))
            df['year'] = pd.to_numeric(extracted['year'], errors='coerce').astype('Int64')
            df['quarter'] = pd.to_numeric(extracted['quarter'], errors='coerce').astype('Int64')
            df = df.drop(columns=['period'])
            logger.info(f"Split period → year + quarter ({qtr_hits}/{total} matched)")

        elif yr_hits / total >= threshold:
            # Plain year: "2024" → year=2024
            df['year'] = pd.to_numeric(
                df['period'].astype(str).str.strip(), errors='coerce'
            ).astype('Int64')
            df = df.drop(columns=['period'])
            logger.info(f"Split period → year ({yr_hits}/{total} matched)")

        elif range_hits / total >= threshold:
            # Date range: "April 01, 2018 - May 31, 2018"
            # Parse both start and end of every range, then decide output columns
            def _parse_range(val):
                m = self._DATE_RANGE_RE.match(str(val).strip())
                if not m:
                    return pd.NA, pd.NA, pd.NA, pd.NA
                fm = self._MONTH_MAP.get(m.group(1).lower()[:3])
                fy = int(m.group(3))
                tm = self._MONTH_MAP.get(m.group(4).lower()[:3])
                ty = int(m.group(6))
                return fm, fy, tm, ty

            parsed = df['period'].apply(
                lambda v: pd.Series(_parse_range(v),
                                    index=['_fm', '_fy', '_tm', '_ty'])
            )
            fm = pd.to_numeric(parsed['_fm'], errors='coerce')
            fy = pd.to_numeric(parsed['_fy'], errors='coerce')
            tm = pd.to_numeric(parsed['_tm'], errors='coerce')
            ty = pd.to_numeric(parsed['_ty'], errors='coerce')

            valid = fm.notna()
            same_month = ((fm == tm) & (fy == ty) & valid)
            # Fiscal year: Apr of year Y → Mar of year Y+1
            is_fy = ((fm == 4) & (tm == 3) & (ty == fy + 1) & valid)

            if same_month.all():
                # All ranges are single-month → month + year
                df['month'] = tm.astype('Int64')
                df['year'] = ty.astype('Int64')
                df = df.drop(columns=['period'])
                logger.info(f"Split period (single-month range) → month + year ({range_hits}/{total} matched)")

            elif is_fy.all():
                # All ranges are fiscal years (Apr Y → Mar Y+1) → fiscal_year
                df['fiscal_year'] = fy.astype(int).astype(str) + '-' + (ty % 100).astype(int).apply(lambda v: f"{v:02d}")
                df = df.drop(columns=['period'])
                logger.info(f"Split period (fiscal-year range) → fiscal_year ({range_hits}/{total} matched)")

            else:
                # Mixed / multi-month ranges → from_month, from_year, to_month, to_year
                df['from_month'] = fm.astype('Int64')
                df['from_year'] = fy.astype('Int64')
                df['to_month'] = tm.astype('Int64')
                df['to_year'] = ty.astype('Int64')
                df = df.drop(columns=['period'])
                logger.info(f"Split period (date range) → from_month/year + to_month/year ({range_hits}/{total} matched)")

        else:
            logger.info("Period column has mixed/unrecognised format — keeping as-is")

        return df
    
    def remove_footer_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove footnote/notes rows from the bottom of the DataFrame.
        Scans from the last row upward looking for patterns like
        'Notes:', 'Source:', numbered footnotes, or long single-cell text.
        """
        if df.empty:
            return df

        drop_indices = []
        for idx in reversed(df.index):
            row = df.loc[idx]
            non_null = row.dropna()
            non_empty = non_null[non_null.astype(str).str.strip() != '']

            # Fully empty row — mark for drop and keep scanning
            if len(non_empty) == 0:
                drop_indices.append(idx)
                continue

            first_val = str(non_empty.iloc[0]).strip()

            # Matches footer patterns (Notes:, Source:, numbered footnotes, etc.)
            if _FOOTER_RE.match(first_val):
                drop_indices.append(idx)
                continue

            # Single long text in 1-2 cells = likely a footnote
            if len(non_empty) <= 2 and len(first_val) > 60:
                drop_indices.append(idx)
                continue

            # Not a footer row — stop scanning
            break

        if drop_indices:
            df = df.drop(index=drop_indices)
            logger.info(f"Removed {len(drop_indices)} footer/note rows from bottom")

        return df

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
        
        # Remove mostly-empty rows (>80% NaN) — catches stray separator rows
        if len(df.columns) >= 3:
            thresh = max(int(len(df.columns) * 0.2), 1)  # keep if >= 20% non-null
            df = df.dropna(thresh=thresh)
        
        # Remove footer/footnote rows that may have survived
        df = self.remove_footer_rows(df)
        
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
        # Only match dashes surrounded by spaces (separators), not word-internal hyphens
        name = re.sub(r'\s+[-–—]\s+.*$', '', name)
        
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
        
        # Capture raw column names before any cleaning (for unit extraction)
        raw_column_names = [str(c) for c in df.columns]
        
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

        # Step 3.5: Split explicit period column for files already in long format
        # (e.g., CSVs with a native `period` column and no date-column melt step)
        if 'period' in df.columns:
            before_cols = set(df.columns)
            df = self._split_period_to_columns(df)
            after_cols = set(df.columns)
            if 'period' not in after_cols and 'period' in before_cols:
                summary_steps.append("Split period into structured time columns")
        
        # Step 4: Add standard columns (refresh_date, data_period when determinable)
        df["refresh_date"] = date.today().isoformat()
        data_period = analysis.get("data_period")
        if data_period:
            df["data_period"] = self._normalize_mmm_yyyy(data_period)
        
        # Step 5: Extract unit info from raw column names into separate columns
        df = self._extract_units(df, raw_column_names)
        
        # Step 6: Add month_numeric column if derivable
        df = self._add_month_numeric(df)
        
        # Check if any columns were dropped
        dropped_cols = original_col_count - len(df.columns)
        if dropped_cols > 0:
            summary_steps.append(f"Dropped {dropped_cols} empty columns")
        
        summary_steps.append("Applied data cleaning (whitespace, empty rows, column names)")
        
        summary = " | ".join(summary_steps) if summary_steps else "No preprocessing needed"
        logger.info(f"Preprocessing completed: {summary}")
        
        return df, summary

    # ── Month-name-to-number mapping (full + abbreviated, lowercase) ──
    _MONTH_MAP = {
        'january': 1, 'jan': 1,
        'february': 2, 'feb': 2,
        'march': 3, 'mar': 3,
        'april': 4, 'apr': 4,
        'may': 5,
        'june': 6, 'jun': 6,
        'july': 7, 'jul': 7,
        'august': 8, 'aug': 8,
        'september': 9, 'sep': 9, 'sept': 9,
        'october': 10, 'oct': 10,
        'november': 11, 'nov': 11,
        'december': 12, 'dec': 12,
    }

    def _add_month_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Deterministically add a `month_numeric` column (1-12) when derivable.

        Priority:
          a. Column already exists → skip.
          b. A 'month' column with all-numeric 1-12 values → rename to month_numeric.
          c. A 'month' column with text month names → keep original + add month_numeric.
          d. A 'period' column with parseable dates → derive month_numeric.
          e. A 'month_name' column (no 'month' column) → derive month_numeric.
        """
        cols_lower = {c.lower(): c for c in df.columns}

        # (a) Already present
        if 'month_numeric' in cols_lower:
            logger.info("month_numeric column already exists, skipping derivation")
            return df

        # (b) 'month' column with numeric values 1-12
        if 'month' in cols_lower:
            month_col = cols_lower['month']
            df = self._coalesce_duplicate_named_columns(df, month_col)
            try:
                numeric_vals = pd.to_numeric(df[month_col], errors='coerce')
                if numeric_vals.dropna().between(1, 12).all() and numeric_vals.notna().sum() > 0:
                    df['month_numeric'] = numeric_vals.astype('Int64')
                    logger.info(f"Derived month_numeric from numeric '{month_col}' column")
                    return df
            except Exception:
                pass

            # (c) 'month' column with text month names
            text_vals = df[month_col].dropna().astype(str).str.strip().str.lower()
            mapped = text_vals.map(self._MONTH_MAP)
            if mapped.notna().sum() > 0 and mapped.notna().sum() / max(len(mapped), 1) > 0.5:
                df['month_numeric'] = df[month_col].astype(str).str.strip().str.lower().map(self._MONTH_MAP)
                logger.info(f"Derived month_numeric from text '{month_col}' column")
                return df

        # (d) 'period' column with parseable dates
        if 'period' in cols_lower:
            period_col = cols_lower['period']
            df = self._coalesce_duplicate_named_columns(df, period_col)
            try:
                parsed = pd.to_datetime(df[period_col], errors='coerce')
                if parsed.notna().sum() / max(len(parsed), 1) > 0.5:
                    df['month_numeric'] = parsed.dt.month
                    logger.info(f"Derived month_numeric from '{period_col}' column")
                    return df
            except Exception:
                pass
            # Try regex extraction for formats like "Jan-2023", "Feb 2023"
            try:
                month_str = df[period_col].astype(str).str.strip().str.lower()
                month_str = month_str.str.extract(r'^([a-z]+)', expand=False)
                mapped = month_str.map(self._MONTH_MAP)
                if mapped.notna().sum() / max(len(df), 1) > 0.5:
                    df['month_numeric'] = mapped
                    logger.info(f"Derived month_numeric from '{period_col}' via regex")
                    return df
            except Exception:
                pass

        # (e) 'month_name' column (no 'month' column)
        if 'month_name' in cols_lower and 'month' not in cols_lower:
            mn_col = cols_lower['month_name']
            df = self._coalesce_duplicate_named_columns(df, mn_col)
            text_vals = df[mn_col].dropna().astype(str).str.strip().str.lower()
            mapped = text_vals.map(self._MONTH_MAP)
            if mapped.notna().sum() > 0:
                df['month_numeric'] = df[mn_col].astype(str).str.strip().str.lower().map(self._MONTH_MAP)
                logger.info(f"Derived month_numeric from '{mn_col}' column")
                return df

        logger.info("No month source found — month_numeric not added")
        return df

    def _extract_units(self, df: pd.DataFrame, raw_column_names: List[str]) -> pd.DataFrame:
        """
        For columns whose raw name contained parenthetical unit info
        (e.g. "Quantity (000 Metric Tonnes)"), add a sibling `<col>_unit` column
        with the constant unit string.
        """
        # Build mapping: cleaned_name → unit_string  from raw names
        unit_pattern = re.compile(r'\(([^)]+)\)')
        # Keywords that are NOT units (structural/formula markers)
        non_unit_keywords = {'adjusted', 'unadjusted', 'total', 'net', 'gross',
                             'actual', 'estimated', 'provisional', 'revised',
                             'final', 'preliminary'}

        raw_to_unit = {}
        for raw in raw_column_names:
            matches = unit_pattern.findall(raw)
            for m in matches:
                m_stripped = m.strip()
                # Skip formulas like "2+3", pure numbers, and non-unit keywords
                if re.match(r'^[\d+\-*/().]+$', m_stripped):
                    continue
                if m_stripped.lower() in non_unit_keywords:
                    continue
                # This looks like a unit — map raw name → unit
                cleaned = self._clean_column_name(raw)
                raw_to_unit[cleaned] = m_stripped

        if not raw_to_unit:
            return df

        for cleaned_col, unit_str in raw_to_unit.items():
            if cleaned_col in df.columns:
                unit_col_name = f"{cleaned_col}_unit"
                if unit_col_name not in df.columns:
                    df[unit_col_name] = unit_str
                    logger.info(f"Added unit column '{unit_col_name}' = '{unit_str}'")

        return df

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
