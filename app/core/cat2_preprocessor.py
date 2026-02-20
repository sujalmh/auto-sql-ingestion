from typing import Dict, List, Tuple, Optional
import pandas as pd
import re
import json
from datetime import date
from openai import OpenAI

from app.config import settings
from app.core.logger import logger
from app.core.preprocessor import preprocessor


class Cat2Preprocessor:
    """
    Preprocessor for Category 2 (lightly structured) tables: row classification,
    hierarchy resolution, subtotal/footer removal, Indian number parsing,
    wide-to-long flattening, and standard columns.
    """

    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model

    def preprocess(
        self, df: pd.DataFrame, analysis: Dict, classification: Dict
    ) -> Tuple[pd.DataFrame, str]:
        """
        Main pipeline. Returns (processed_df, summary_string).

        CRITICAL: Input df already has resolved headers from Cat1 logic (Step 1-2).
        We run merge_headers first so column mapping matches Cat1, then row logic.
        """
        summary_steps = []

        # 0. Same header merge as Cat1 so cells map to correct columns
        df = preprocessor.merge_headers(df, analysis)
        if analysis.get("needs_header_merge"):
            summary_steps.append("Merged multi-level headers")

        # 1. Identify structural columns
        index_col, label_col = self.identify_structural_columns(df, analysis)

        # 2. Classify all rows via LLM
        row_classifications = self.classify_all_rows(df, index_col, label_col)

        # 3. Resolve hierarchy (if present)
        if row_classifications.get("hierarchy_present"):
            df = self.resolve_hierarchy(df, row_classifications)
            depth = row_classifications.get("hierarchy_depth", 0)
            summary_steps.append(f"Resolved {depth}-level hierarchy")

        # 4. Filter to data rows only
        total_before = len(df)
        df = self.filter_data_rows(df, row_classifications)
        removed = total_before - len(df)
        summary_steps.append(f"Removed {removed} non-data rows")

        # 5. Drop structural index column
        if index_col and index_col in df.columns:
            df = df.drop(columns=[index_col])
            summary_steps.append(f"Dropped structural column: {index_col}")

        # 6. Standardize labels
        df = self.standardize_labels(df)
        summary_steps.append("Standardized labels")

        # 7. Parse Indian numbers
        df = self.parse_indian_numbers(df)

        # 8. Flatten wide-to-long (if applicable)
        date_columns = analysis.get("date_columns", [])
        if date_columns:
            df = self.flatten_wide_to_long(df, analysis)
            summary_steps.append("Flattened wide-to-long")

        # 9. Add standard columns
        df = self.add_standard_columns(df, analysis)
        summary_steps.append("Added refresh_date")

        # 10. Reuse Cat1 clean_data
        df = preprocessor.clean_data(df)

        summary = " | ".join(summary_steps)
        return df, summary

    def identify_structural_columns(
        self, df: pd.DataFrame, analysis: Dict
    ) -> Tuple[Optional[str], str]:
        """Identify index column and label column. Returns (index_col_or_None, label_col)."""
        columns = df.columns.tolist()
        if isinstance(df.columns, pd.MultiIndex):
            columns = [str(c) for c in df.columns]

        first_3 = df.head(3)
        rows_repr = []
        for i, row in first_3.iterrows():
            vals = [str(v) if pd.notna(v) else "NaN" for v in row.tolist()]
            rows_repr.append(f"Row {i}: {vals}")

        prompt = f"""You are analyzing a table to identify structural columns.

Column names: {columns}

First 3 rows:
{chr(10).join(rows_repr)}

Tasks:
1. Identify the INDEX column (contains hierarchy markers like A/I/i, 1/1.1/1.1.1, or section numbers). Return null if no such column exists.
2. Identify the LABEL column (contains text descriptions of items/categories).

Return JSON only, with keys: "index_column" (string or null), "label_column" (string).
Example: {{"index_column": "Sr", "label_column": "Scheme Name"}}
If no index: {{"index_column": null, "label_column": "States/UTs"}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You identify structural columns in data tables. Respond only with valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            result = json.loads(response.choices[0].message.content)
            index_col = result.get("index_column")
            if index_col is None or (isinstance(index_col, str) and index_col.lower() == "null"):
                index_col = None
            label_col = result.get("label_column")
            if not label_col or label_col not in df.columns:
                # Fallback: first string-like column
                for c in df.columns:
                    if df[c].dtype == object or str(df[c].dtype) == "object":
                        label_col = c
                        break
                else:
                    label_col = columns[0] if columns else ""
            return (index_col, label_col)
        except Exception as e:
            logger.error(f"identify_structural_columns failed: {e}")
            for c in df.columns:
                if df[c].dtype == object or str(df[c].dtype) == "object":
                    return (None, c)
            return (None, columns[0] if columns else "")

    def classify_all_rows(
        self,
        df: pd.DataFrame,
        index_col: Optional[str],
        label_col: str,
    ) -> Dict:
        """Send index+label of ALL rows to LLM. Returns full classification dict."""
        rows_text = []
        for i, row in df.iterrows():
            idx_val = str(row[index_col]).strip() if index_col and index_col in df.columns else ""
            label_val = str(row[label_col]).strip() if pd.notna(row.get(label_col)) else ""
            rows_text.append(f'Row {i}: Index="{idx_val}", Label="{label_val}"')

        all_rows_str = "\n".join(rows_text)

        prompt = f"""You are analyzing a data table to classify each row's structural role.

Here are the index and label columns for every row:

{all_rows_str}

Classify each row into one of these roles:
- "data": An actual data row with real values to keep
- "group_header": A section header that labels a group of rows. Include hierarchy level (0=top, 1=next, etc.)
- "subtotal": A total/subtotal/aggregation row
- "footer": A note, source citation, disclaimer, or non-data text at the bottom
- "derived": A percentage/ratio/calculated summary row (e.g., "Distribution %")

Return JSON only:
{{
  "rows": [
    {{"row": 0, "role": "group_header", "level": 0, "label": "..."}},
    {{"row": 1, "role": "group_header", "level": 1, "label": "..."}},
    {{"row": 2, "role": "data"}},
    ...
  ],
  "hierarchy_present": true or false,
  "hierarchy_depth": number of levels (0 if no hierarchy),
  "group_columns_to_create": ["category", "sub_category"] or similar names for hierarchy columns
}}
For "data" rows do not include "level" or "label". For "group_header" include "level" and "label"."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You classify table rows as data, group_header, subtotal, footer, or derived. Respond only with valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            result = json.loads(response.choices[0].message.content)
            rows = result.get("rows", [])
            data_count = sum(1 for r in rows if r.get("role") == "data")
            header_count = sum(1 for r in rows if r.get("role") == "group_header")
            subtotal_count = sum(1 for r in rows if r.get("role") == "subtotal")
            footer_count = sum(1 for r in rows if r.get("role") == "footer")
            logger.info(
                f"Row classification: {data_count} data, {header_count} headers, "
                f"{subtotal_count} subtotals, {footer_count} footers"
            )
            return result
        except Exception as e:
            logger.error(f"classify_all_rows failed: {e}")
            # Default: all rows are data
            return {
                "rows": [{"row": i, "role": "data"} for i in range(len(df))],
                "hierarchy_present": False,
                "hierarchy_depth": 0,
                "group_columns_to_create": [],
            }

    def resolve_hierarchy(self, df: pd.DataFrame, row_classifications: Dict) -> pd.DataFrame:
        """Add hierarchy columns to data rows based on group_header labels."""
        parent_labels = {}
        group_cols = row_classifications.get("group_columns_to_create", [])

        for row_info in row_classifications.get("rows", []):
            idx = row_info["row"]
            role = row_info.get("role")

            if role == "group_header":
                level = row_info.get("level", 0)
                label = row_info.get("label", "")
                parent_labels[level] = label
                for deeper in list(parent_labels):
                    if deeper > level:
                        del parent_labels[deeper]

            elif role == "data":
                for level_idx, col_name in enumerate(group_cols):
                    df.at[idx, col_name] = parent_labels.get(level_idx, None)

        return df

    def filter_data_rows(self, df: pd.DataFrame, row_classifications: Dict) -> pd.DataFrame:
        """Keep only rows classified as 'data'."""
        data_indices = [
            r["row"] for r in row_classifications.get("rows", []) if r.get("role") == "data"
        ]
        if not data_indices:
            return df.iloc[0:0].copy()
        return df.loc[data_indices].reset_index(drop=True)

    def standardize_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Strip whitespace, asterisks, footnote markers from string columns."""
        for i in range(len(df.columns)):
            ser = df.iloc[:, i]
            if ser.dtype != object and str(ser.dtype) != "object":
                continue
            s = ser.astype(str)
            s = s.str.strip()
            s = s.str.replace(r"\s*\*+\s*$", "", regex=True)
            s = s.str.replace(r"\s*[@#†‡§]+\s*$", "", regex=True)
            s = s.str.replace(r"\s*\(\d+\)\s*$", "", regex=True)
            df.iloc[:, i] = s.values
        return df

    def parse_indian_numbers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert Indian-format numbers (1,23,456.78) to numeric."""
        indian_pattern = re.compile(r"^\d{1,2}(,\d{2})*(,\d{3})?(\.\d+)?$|^\d+(\.\d+)?$")

        for i in range(len(df.columns)):
            ser = df.iloc[:, i]
            if ser.dtype not in (object, "object"):
                continue
            non_null = ser.dropna().astype(str).str.strip()
            non_null = non_null[non_null != ""]
            if len(non_null) == 0:
                continue
            matches = non_null.str.replace(",", "").str.match(r"^-?\d+\.?\d*$") | non_null.str.match(
                indian_pattern
            )
            if matches.sum() / len(non_null) < 0.5:
                continue
            # Convert: remove commas, handle dash as NaN
            def to_num(val):
                if pd.isna(val) or str(val).strip() in ("", "-", "—", "–"):
                    return pd.NA
                return str(val).replace(",", "")

            ser = ser.apply(to_num)
            df.iloc[:, i] = pd.to_numeric(ser, errors="coerce").values
        return df

    def flatten_wide_to_long(self, df: pd.DataFrame, analysis: Dict) -> pd.DataFrame:
        """Melt date/period columns if analysis indicates wide format. Skip when columns are metric names (e.g. AMFI)."""
        date_columns = analysis.get("date_columns", [])
        if not date_columns:
            return df
        actual_date_cols = [c for c in date_columns if c in df.columns]
        if not actual_date_cols:
            for dc in date_columns:
                matches = [c for c in df.columns if dc in str(c)]
                actual_date_cols.extend(matches)
        if not actual_date_cols:
            return df
        # Do not melt when "date" columns look like metric descriptions (e.g. "Funds Mobilised for the month of Dec 2025")
        metric_like = re.compile(
            r"crore|INR|for the month|as on|No\. of|Funds Mobilised|Net Assets|"
            r"Repurchase|Redemption|Inflow|Outflow|Average Net|segregated portfolio",
            re.I,
        )
        if any(len(c) > 50 or metric_like.search(c) for c in actual_date_cols):
            logger.info(
                "Skipping wide-to-long: matched columns look like metric names, not period columns"
            )
            return df
        id_cols = [c for c in df.columns if c not in actual_date_cols]
        if not id_cols:
            df = df.copy()
            df["_id"] = range(len(df))
            id_cols = ["_id"]
        df = pd.melt(
            df,
            id_vars=id_cols,
            value_vars=actual_date_cols,
            var_name="period",
            value_name="value",
        )
        if "_id" in id_cols:
            df = df.drop(columns=["_id"])
        return df

    def add_standard_columns(self, df: pd.DataFrame, analysis: Dict) -> pd.DataFrame:
        """Add refresh_date, data_period (Mmm-yyyy e.g. Dec 2025 when determinable), and optional period/fy from period_info."""
        df = df.copy()
        df["refresh_date"] = date.today().isoformat()

        # Data period: time of data (e.g. Dec 2025) from analysis
        data_period = analysis.get("data_period")
        if data_period:
            df["data_period"] = self._normalize_mmm_yyyy(data_period)

        period_info = analysis.get("period_info", {})
        if period_info.get("detected") and period_info.get("add_as_column"):
            pval = period_info.get("period_value")
            if pval:
                df["period"] = pval
                fy = self._derive_fy(pval)
                if fy:
                    df["fy"] = fy

        return df

    def _normalize_mmm_yyyy(self, value: str) -> str:
        """Convert period string to Mmm-yyyy (e.g. Dec 2025)."""
        if not value or not str(value).strip():
            return value
        value = str(value).strip()
        month_abbr = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
        # Already MM-YYYY or M-YYYY
        if re.match(r"^\d{1,2}-\d{4}$", value):
            m, y = value.split("-")
            return f"{month_abbr[int(m) - 1]} {y}"
        # YYYY-MM
        if re.match(r"^\d{4}-\d{1,2}$", value):
            y, m = value.split("-")
            return f"{month_abbr[int(m) - 1]} {y}"
        # Try to extract month and year
        year_m = re.search(r"\b(20\d{2})\b", value)
        months_lower = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
        }
        month_num = None
        for name, num in months_lower.items():
            if name in value.lower():
                month_num = num
                break
        if year_m and month_num is not None:
            return f"{month_abbr[month_num - 1]} {year_m.group(1)}"
        return value

    def _derive_fy(self, period_value: str) -> Optional[str]:
        """Derive Indian fiscal year from period string. April–March."""
        if not period_value:
            return None
        # Try common patterns: "December 2025", "Dec 2025", "2025", "Q1-2024", "Apr-2023"
        import re
        period_value = str(period_value).strip()
        year = None
        month = None
        # Year: 4-digit
        year_m = re.search(r"\b(20\d{2})\b", period_value)
        if year_m:
            year = int(year_m.group(1))
        # Month name
        months = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
        }
        for name, num in months.items():
            if name in period_value.lower():
                month = num
                break
        if year is None:
            return None
        if month is None:
            return f"FY{year}-{year + 1}"
        if month >= 4:
            return f"FY{year}-{year + 1}"
        return f"FY{year - 1}-{year}"

    def build_output_schema(
        self, df: pd.DataFrame, row_classifications: Dict
    ) -> Dict:
        """LLM decides final column names, dimensions vs measures, columns to drop."""
        columns = df.columns.tolist()
        prompt = f"""Given these table columns after row filtering: {columns}

Identify:
1. dimension_columns: text/label columns (original name -> clean_name for PostgreSQL)
2. measure_columns: numeric columns (original -> clean_name, dtype INTEGER or NUMERIC)
3. columns_to_drop: structural columns to remove (e.g. index/Sr)

Return JSON:
{{
  "dimension_columns": [{{"original": "...", "clean_name": "..."}}],
  "measure_columns": [{{"original": "...", "clean_name": "...", "dtype": "INTEGER"}}],
  "columns_to_drop": ["Sr"],
  "column_rename_map": {{"Original": "clean_name", ...}}
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You define output schema for SQL tables. Respond only with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.warning(f"build_output_schema failed: {e}")
            return {
                "dimension_columns": [],
                "measure_columns": [],
                "columns_to_drop": [],
                "column_rename_map": {},
            }


cat2_preprocessor = Cat2Preprocessor()
