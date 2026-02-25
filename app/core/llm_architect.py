from openai import OpenAI
import pandas as pd
import json
from typing import Any, Dict, List, Optional, Tuple
from app.config import settings
from app.core.logger import logger


class LLMArchitect:
    """
    LLM-powered intelligent agent for analyzing data files and making decisions
    about table naming, preprocessing strategies, and schema inference.
    """
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
        logger.info(f"LLMArchitect initialized with model: {self.model}")
    
    def analyze_file_structure(self, df: pd.DataFrame) -> Dict:
        """
        Analyze the DataFrame structure to identify preprocessing needs.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing analysis results
        """
        logger.info("Analyzing file structure with LLM")
        
        # Prepare file information for LLM
        # Include first few rows to help detect multi-level headers
        
        # Handle MultiIndex columns - create a temporary copy with string column names
        # to avoid JSON serialization errors with tuple keys
        df_for_analysis = df.copy()
        has_multiindex = isinstance(df.columns, pd.MultiIndex)
        
        if has_multiindex:
            # Convert MultiIndex to string representation for JSON serialization
            df_for_analysis.columns = [str(col) for col in df.columns]
            logger.debug("Converted MultiIndex columns to strings for LLM analysis")
        
        file_info = {
            "shape": df.shape,
            "columns": df.columns.tolist() if not has_multiindex else [str(col) for col in df.columns],
            "first_10_rows": df_for_analysis.head(10).to_dict(orient='records'),
            "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
            "has_multiindex": has_multiindex
        }
        
        prompt = f"""Analyze this Excel/CSV file structure and identify preprocessing needs.

File Information:
- Shape: {file_info['shape'][0]} rows × {file_info['shape'][1]} columns
- Has MultiIndex columns: {file_info['has_multiindex']}
- Columns: {file_info['columns']}
- First 10 rows sample: {json.dumps(file_info['first_10_rows'], indent=2, default=str)}

Tasks:
1. date_columns: List ONLY columns whose HEADER VALUE is a period/year (e.g. "2011-12", "2012-13", "Q1", "January") for wide-to-long melt. Leave date_columns EMPTY if columns are metric names that merely mention a date in the description (e.g. "Funds Mobilised for the month of Dec 2025 (INR in crore)" or "No. of Schemes as on Dec 31, 2025")—those are single-point metrics, not time-series columns to melt.
2. Detect if headers span multiple levels and need merging. This includes:
   - Pandas MultiIndex columns
   - First few rows containing header information (e.g., "Unit value index" spanning multiple columns)
   - Look for patterns where first rows have duplicate values or many NaN values
3. Identify the grain/dimensionality of the data using canonical abbreviations (all lowercase):
   geo: india, state, dist | time: dly, wk, mth, qtr, yr | grain: catg, sctg, sector, commodity, industry, instrument, scheme, bank
   Example: "state-mth-catg" for state-wise monthly data with category breakdown.
4. Suggest preprocessing strategy
5. data_period: When column names or context indicate a single reference date for the data (e.g. "as on Dec 31, 2025", "for the month of Dec 2025", "December 2025"), extract it and return in Mmm-yyyy format (e.g. "Dec 2025"). Use for the final table so each row has the "time of data". Omit or null if not determinable.

IMPORTANT for needs_header_merge:
- Set needs_header_merge TRUE only when MULTIPLE ROWS together form the COLUMN NAMES (e.g. row 0 = "Category", row 1 = "Year | State | Value" with labels in MANY columns).
- Set needs_header_merge FALSE when the first rows have text only in the FIRST 1-2 columns and the rest empty/NaN. Those are SECTION or CATEGORY title rows (e.g. "A" / "Open ended Schemes", "I" / "Income/Debt..."), not column header rows. The table already has one row of column names; do not merge section rows into headers.

Respond in JSON format:
{{
    "has_date_headers": true/false,
    "date_columns": ["list of column names with dates"],
    "needs_header_merge": true/false,
    "header_merge_reason": "explanation if needs_header_merge is true",
    "data_grain": "description of data granularity",
    "domain": "data domain (e.g., IIP, GDP, Trade, etc.)",
    "preprocessing_strategy": "brief description of recommended preprocessing",
    "data_period": "Mmm-yyyy e.g. Dec 2025 when data has a single reference date from column names; omit if not determinable"
}}"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a data analysis expert. Analyze data structures and provide preprocessing recommendations in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            analysis = json.loads(response.choices[0].message.content)
            logger.info(f"File structure analysis completed: {analysis}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error in file structure analysis: {str(e)}")
            # Return default analysis on error
            return {
                "has_date_headers": False,
                "date_columns": [],
                "needs_header_merge": False,
                "header_merge_reason": "",
                "data_grain": "unknown",
                "domain": "unknown",
                "preprocessing_strategy": "standard cleaning",
                "data_period": None,
            }
    
    def generate_table_name(self, df: pd.DataFrame, analysis: Dict, file_description: str = None) -> str:
        """
        Generate a PostgreSQL table name based on file analysis.
        
        Args:
            df: DataFrame to analyze
            analysis: Analysis results from analyze_file_structure
            file_description: Optional user-provided description to help with naming
            
        Returns:
            Generated table name
        """
        logger.info("Generating table name with LLM")
        
        naming_convention = """
Naming Convention:  <domain>_<geo>_<time>_<grain_dimension>
ALL segments MUST be lowercase snake_case.

  - <domain>   (mandatory) — data domain. Use well-known abbreviations when they
                exist (gdp, cpi, iip, gst, msme, fdi, upi, epfo …); otherwise
                use the full descriptive name in snake_case.
  - <geo>      (optional)  — geographic level of the data.
  - <time>     (optional)  — time granularity of the data.
  - <grain_dimension> (optional) — the lowest-level dimension / grain of rows.

Canonical Abbreviation Map (use these consistently, all lowercase):
  geo   : india, state, dist (district), global, region
  time  : dly (daily), wk (weekly), mth (monthly), qtr (quarterly), yr (yearly)
  grain : catg (category), sctg (sub-category), sector, commodity, industry,
          instrument, scheme, bank, fund, company, product

Examples across domains (all lowercase snake_case):
  iip_india_mth_sctg        — IIP India Monthly at Sub-Category grain
  iip_state_mth_catg        — IIP State-wise Monthly at Category grain
  iip_state_mth_sector      — IIP State-wise Monthly at Sector grain
  cpi_india_mth_commodity   — CPI India Monthly at Commodity grain
  cpi_state_yr_catg         — CPI State-wise Yearly at Category grain
  gdp_india_qtr_sector      — GDP India Quarterly at Sector grain
  gdp_india_yr_industry     — GDP India Yearly at Industry grain
  trade_india_mth_commodity — Trade India Monthly at Commodity grain
  msme_india_yr_industry    — MSME India Yearly at Industry grain
  gst_state_mth_sector      — GST State-wise Monthly at Sector grain
  mutual_fund_india_mth_scheme — Mutual Fund India Monthly at Scheme grain
  fdi_india_yr_sector       — FDI India Yearly at Sector grain
  upi_india_mth_bank        — UPI India Monthly at Bank grain
  rainfall_dist_mth         — Rainfall District-wise Monthly (no further grain)
  epfo_state_mth_industry   — EPFO State-wise Monthly at Industry grain
"""
        
        # Handle MultiIndex columns for JSON serialization
        df_for_analysis = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df_for_analysis.columns = [str(col) for col in df.columns]
        
        sample_data = df_for_analysis.head(10).to_dict(orient='records')
        column_list = df.columns.tolist() if not isinstance(df.columns, pd.MultiIndex) else [str(col) for col in df.columns]
        
        # Build prompt with optional file description
        description_context = ""
        if file_description:
            description_context = f"\n\nUser-provided context: {file_description}\nUse this to better understand the data and generate a more meaningful table name."
        
        prompt = f"""Generate a PostgreSQL table name following the naming convention.

{naming_convention}{description_context}

Data Analysis:
- Domain: {analysis.get('domain', 'unknown')}
- Data Grain: {analysis.get('data_grain', 'unknown')}
- Columns: {column_list}
- Sample rows (first 10): {json.dumps(sample_data, indent=2, default=str)}

Generate an appropriate table name following the convention strictly.
The name MUST be all lowercase snake_case (e.g. iip_india_mth_sctg, NOT IIP_India_Mth_SCtg).
Use the canonical abbreviation map above — do NOT invent your own abbreviations.
Omit optional segments only when the data truly has no geographic, temporal,
or dimensional breakdown.

Respond in JSON format:
{{
    "table_name": "<domain>_<geo>_<time>_<grain_dimension>",
    "reasoning": "brief explanation of the naming choice"
}}"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a database naming expert specializing in Indian economic and statistical datasets. Generate table names following strict naming conventions and canonical abbreviations."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            result = json.loads(response.choices[0].message.content)
            table_name = result.get("table_name", "DATA_TABLE")
            reasoning = result.get("reasoning", "")
            
            logger.info(f"Generated table name: {table_name} (Reasoning: {reasoning})")
            return table_name
            
        except Exception as e:
            logger.error(f"Error generating table name: {str(e)}")
            return "DATA_TABLE_DEFAULT"
    
    def infer_column_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Infer PostgreSQL column types for the DataFrame.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary mapping column names to PostgreSQL types
        """
        logger.info("Inferring PostgreSQL column types with LLM")
        
        # Get sample data and pandas dtypes
        column_info = []
        for col in df.columns:
            # Safely access column data
            col_data = df[col]
            if isinstance(col_data, pd.DataFrame):
                # If df[col] returns DataFrame (duplicate column names), take first
                col_data = col_data.iloc[:, 0]
            
            # Get sample values
            sample_values = col_data.dropna().head(10).tolist()
            
            # For string columns, calculate max length
            max_length = None
            if col_data.dtype == 'object':
                try:
                    max_length = col_data.astype(str).str.len().max()
                except:
                    max_length = None
            
            column_info.append({
                "name": col,
                "pandas_dtype": str(col_data.dtype),
                "sample_values": sample_values,
                "null_count": int(col_data.isna().sum()),
                "total_count": len(col_data),
                "max_string_length": max_length
            })
        
        prompt = f"""Infer PostgreSQL column types for the following columns.

Column Information:
{json.dumps(column_info, indent=2, default=str)}

Rules:
1. For text columns, use the max_string_length to determine VARCHAR size
   - If max_string_length > 255, use TEXT instead of VARCHAR
   - If max_string_length <= 50, use VARCHAR(100) for safety margin
   - If max_string_length <= 100, use VARCHAR(200) for safety margin
   - If max_string_length <= 255, use VARCHAR(500) for safety margin
2. For numeric columns, infer precision and scale from sample values
3. For date/time columns, use TIMESTAMP or DATE as appropriate
4. Use TEXT for long descriptive fields

Return JSON with column names as keys and PostgreSQL types as values.
Example: {{"column_name": "VARCHAR(200)", "amount": "NUMERIC(15, 2)", "date": "TIMESTAMP"}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a database schema expert. Infer optimal PostgreSQL data types from sample data."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Validate and fix VARCHAR sizes to prevent truncation
            for col in df.columns:
                col_data = df[col]
                if isinstance(col_data, pd.DataFrame):
                    col_data = col_data.iloc[:, 0]
                
                # For text columns, ensure VARCHAR is large enough
                if col_data.dtype == 'object' and col in result:
                    inferred_type = result[col]
                    
                    # Check if it's a VARCHAR type
                    if 'VARCHAR' in inferred_type.upper():
                        try:
                            # Extract the size from VARCHAR(n)
                            import re
                            match = re.search(r'VARCHAR\((\d+)\)', inferred_type, re.IGNORECASE)
                            if match:
                                varchar_size = int(match.group(1))
                                max_length = col_data.astype(str).str.len().max()
                                
                                # If max length exceeds VARCHAR size, fix it
                                if max_length > varchar_size:
                                    if max_length > 255:
                                        # Use TEXT for very long strings
                                        result[col] = 'TEXT'
                                        logger.warning(f"Column '{col}': Changed VARCHAR({varchar_size}) to TEXT (max length: {max_length})")
                                    else:
                                        # Use a larger VARCHAR with safety margin
                                        new_size = min(max_length * 2, 500)  # 2x with cap at 500
                                        result[col] = f'VARCHAR({new_size})'
                                        logger.warning(f"Column '{col}': Increased VARCHAR({varchar_size}) to VARCHAR({new_size}) (max length: {max_length})")
                        except Exception as e:
                            logger.warning(f"Could not validate VARCHAR size for '{col}': {e}")
            
            logger.info(f"Inferred column types: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error inferring column types: {str(e)}")
            # Fallback to basic type inference
            return {col: "TEXT" for col in df.columns}
    
    def detect_header_rows(self, df_head: pd.DataFrame) -> int:
        """Analyze first 5 rows to determine header row count."""
        logger.info("Detecting header row count with LLM")
        try:
            markdown = df_head.to_markdown(index=False)
            prompt = f"""Analyze this Excel/CSV data preview and count ONLY the true column-header rows.

Data Preview:
{markdown}

DEFINITION OF A COLUMN-HEADER ROW:
- A row that provides the NAME OF EACH COLUMN (e.g. "Sr", "Scheme Name", "No. of Schemes", "Amount").
- Such a row has NON-EMPTY or MEANINGFUL labels in MANY columns (at least 3–4+ columns across the row).

NOT COLUMN HEADERS (do NOT count these):
- Section or category title rows: only the FIRST 1–2 columns have text, the REST are empty or dashes.
  Example: row with "A" and "Open ended Schemes" in first two cells and empty elsewhere = section header, NOT a column header.
- Rows with roman numerals (I, II, III) or single letters (A, B) in the first column and a category name in the second, with other columns empty = structural/section rows. Return 1 so only the real column title row is used.
- Subtotal rows, data rows, or footer rows.

RULES:
- If row 0 has labels in many columns (true column titles) and row 1 has only 1–2 cells filled (e.g. section title) → return 1.
- Only return 2 or 3 when you see TWO or THREE consecutive rows that each have labels spanning MANY columns (e.g. merged Excel headers where row 0 = "Category", row 1 = "Year | State | Value" with values in many columns).

Respond with ONLY a single number: 1, 2, or 3"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0  # Reduced from 0.1 for more deterministic output
            )
            header_count = int(response.choices[0].message.content.strip())
            if header_count not in [1, 2, 3]:
                logger.warning(f"LLM returned invalid header count: {header_count}, defaulting to 1")
                header_count = 1
            logger.info(f"Detected {header_count} header row(s)")
            return header_count
        except Exception as e:
            logger.error(f"Error detecting headers: {str(e)}, defaulting to 1")
            return 1
    
    def refine_table_name(self, table_name: str) -> str:
        """Shorten table name if too long."""
        if len(table_name) <= 40:
            return table_name
        logger.info(f"Refining long table name: {table_name}")
        try:
            prompt = f"""Shorten this table name to max 40 characters while preserving the
naming convention: <domain>_<geo>_<time>_<grain_dimension>.
The result MUST be all lowercase snake_case.

Original: {table_name}

Use these canonical abbreviations (all lowercase):
  geo  : india, state, dist
  time : dly, wk, mth, qtr, yr
  grain: catg, sctg, sector, commodity, industry, instrument, scheme, bank

Rules:
- Keep all four segments if possible; shorten individual words using the map above
- Use underscores between segments
- Preserve the domain abbreviation exactly as-is (gdp, cpi, iip, etc.)
- Output MUST be all lowercase

Return ONLY the shortened name, nothing else."""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            refined = response.choices[0].message.content.strip().lower()
            if len(refined) > 50:
                refined = '_'.join(refined.split('_')[:4])
            logger.info(f"Refined to: {refined}")
            return refined
        except Exception as e:
            logger.error(f"Error refining name: {str(e)}")
            return table_name[:40]
    
    def clean_column_names(
        self,
        columns: list,
        sample_rows: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Clean and normalize column names; detect redundant columns to drop.
        Returns {"mapping": {orig: renamed, ...}, "drop_columns": [orig, ...]}.
        """
        logger.info(f"Cleaning {len(columns)} column names")
        try:
            prompt = f"""You are normalizing column names for a database table. Your goals:
1. Produce clear, semantic snake_case names (lowercase, underscores, max 50 chars).
2. Remove redundant columns: when two or more columns carry the same semantic meaning or identical values, keep ONE with a canonical name and list the others in "drop_columns".
3. Optionally drop metadata-only columns (e.g. ingestion timestamp, refresh_date) if they add no analytical value; otherwise keep with a clear name.

INPUT COLUMNS:
{json.dumps(columns, indent=2)}
"""
            if sample_rows:
                prompt += f"""
SAMPLE ROW DATA (use to detect redundant columns—e.g. same values in two columns = duplicate):
{json.dumps(sample_rows[:3], indent=2, default=str)}
"""
            prompt += """
NAMING RULES (apply to any file; do not hardcode for one dataset):
- Strip numeric suffixes that denote duplicate/copy columns (e.g. year_2, year_2_1 → both mean "year"; keep one as "year", add the other to drop_columns).
- Remove dates, timestamps, or large numbers appended to names (e.g. "_2025-05-30", "_23117503.01").
- Remove special characters: (), [], etc. Replace spaces with underscores.
- Use semantic names: prefer meaning over raw labels (e.g. "india_news" → "india_policy_uncertainty_index" or "policy_uncertainty_index" if context is clear; "description_english" → "description").
- If two columns are semantically different, preserve what distinguishes them (e.g. adjusted vs unadjusted, total vs net).
- Every output name in "mapping" MUST be unique.

STEP-BY-STEP PROCESS:
1. Remove dates and numeric values (e.g., "_2025-05-30", "_23117503.01")
2. Remove special characters: (), [], etc.
3. Convert to snake_case, lowercase
4. Identify what makes similar columns DIFFERENT and preserve it.
5. also remove words that are not relevant to the column name eg: "description_english" can be changed to just "description".

EXAMPLES OF CORRECT HANDLING:

Input: "Aggregate deposits (2+3)_23172542.62"
Output: "aggregate_deposits"

Input: "Aggregate deposits (2+3) (Adjusted)_23117503.01"
Output: "aggregate_deposits_adjusted"  ← Note: preserved "adjusted"

Input: "Investment In India (8+9)_6706717.241"
Output: "investment_in_india"

Input: "India News-Based Policy Uncertainty Index"
Output: "india_news_based_policy_uncertainty_index"
(For long descriptive titles like this, always use the full snake_case form; do not abbreviate to fewer words.)

Input: "Investment In India (8+9) (Adjusted)_6660129.361"
Output: "investment_in_india_adjusted"  ← Note: preserved "adjusted"

Input: "Bank Credit (11+12)_18287376.91"
Output: "bank_credit"

Input: "Bank Credit (11+12) (Adjusted)_17888404.07"
Output: "bank_credit_adjusted"  ← Note: preserved "adjusted"

CRITICAL RULES:
- Every output name MUST be unique
- If two inputs are similar, find what makes them different (adjusted, unadjusted, total, net, gross, etc.)
- Preserve these distinguishing words in the output
- Max 50 characters per name
- Use snake_case, lowercase only
- Economics/financial context

FULL INPUT/OUTPUT EXAMPLE:

Input columns: ["Sr No.", "Scheme Name", "No. of Schemes_2025-05-30", "Net Assets Under Management (Rs Crore)", "Net Assets Under Management (Rs Crore) (Adjusted)_23117503.01"]
Output:
{{
  "mapping": {{
    "Sr No.": "sr_no",
    "Scheme Name": "scheme_name",
    "No. of Schemes_2025-05-30": "no_of_schemes",
    "Net Assets Under Management (Rs Crore)": "net_aum_rs_crore",
    "Net Assets Under Management (Rs Crore) (Adjusted)_23117503.01": "net_aum_rs_crore_adjusted"
  }},
  "drop_columns": []
}}

Return ONLY a JSON object with exactly two keys: "mapping" and "drop_columns".
- "mapping": an object where each key is an original column name and each value is the cleaned snake_case name.
- "drop_columns": a list of original column names to drop."""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            raw = response.choices[0].message.content
            if not raw:
                raise ValueError("LLM returned empty column-cleaning response")
            data = json.loads(raw)
            mapping = data.get("mapping", {})
            drop_columns = data.get("drop_columns", [])
            if not isinstance(drop_columns, list):
                drop_columns = []

            # Ensure we have a mapping for every column that is not dropped
            for col in columns:
                if col not in drop_columns and col not in mapping:
                    mapping[col] = col

            # Validate uniqueness of cleaned names
            cleaned_values = [mapping[c] for c in mapping if c not in drop_columns]
            duplicates = [v for v in set(cleaned_values) if cleaned_values.count(v) > 1]
            if duplicates:
                logger.warning(f"LLM returned duplicate names: {duplicates}; applying post-processing")
                kept_cols = [c for c in columns if c not in drop_columns]
                mapping = self._ensure_unique_column_names(kept_cols, {k: mapping[k] for k in kept_cols})

            logger.info(
                f"Cleaned {len(mapping)} column names, drop {len(drop_columns)}: {drop_columns}"
            )
            return {"mapping": mapping, "drop_columns": drop_columns}
        except Exception as e:
            logger.error(f"Error cleaning names: {str(e)}")
            return {"mapping": {col: col for col in columns}, "drop_columns": []}
    
    def _ensure_unique_column_names(self, original_cols: list, mapping: dict) -> dict:
        """Ensure all column names are unique by adding meaningful suffixes."""
        logger.info("Post-processing to ensure unique column names")
        
        # Reverse mapping to find which original columns map to same cleaned name
        cleaned_to_original = {}
        for orig, cleaned in mapping.items():
            if cleaned not in cleaned_to_original:
                cleaned_to_original[cleaned] = []
            cleaned_to_original[cleaned].append(orig)
        
        # Fix duplicates by preserving distinctions from original names
        new_mapping = {}
        for cleaned_name, orig_names in cleaned_to_original.items():
            if len(orig_names) == 1:
                # No duplicate, keep as is
                new_mapping[orig_names[0]] = cleaned_name
            else:
                # Multiple columns map to same name - add distinguishing suffixes
                logger.info(f"Resolving {len(orig_names)} duplicates for '{cleaned_name}': {orig_names}")
                for i, orig in enumerate(orig_names):
                    if i == 0:
                        new_mapping[orig] = cleaned_name
                    else:
                        # Extract distinguishing part from original name
                        suffix = self._extract_distinguishing_suffix(orig, orig_names[0])
                        new_name = f"{cleaned_name}_{suffix}" if suffix else f"{cleaned_name}_{i}"
                        new_mapping[orig] = new_name
                        logger.info(f"  '{orig}' → '{new_name}' (suffix: {suffix or i})")
        
        return new_mapping
    
    def _extract_distinguishing_suffix(self, col1: str, col2: str) -> str:
        """Extract what makes col1 different from col2."""
        import re
        
        col1_lower = col1.lower()
        col2_lower = col2.lower()
        
        # Look for keywords that distinguish columns
        keywords = ['adjusted', 'unadjusted', 'total', 'net', 'gross', 'actual', 'estimated', 
                    'provisional', 'revised', 'final', 'preliminary']
        
        for keyword in keywords:
            if keyword in col1_lower and keyword not in col2_lower:
                return keyword
        
        # Look for numbers in parentheses like (8+9) vs (2+3)
        match1 = re.search(r'\(([^)]+)\)', col1)
        match2 = re.search(r'\(([^)]+)\)', col2)
        if match1 and match2 and match1.group(1) != match2.group(1):
            # Use the formula as suffix, cleaned
            formula = re.sub(r'[^a-z0-9]', '', match1.group(1).lower())
            if formula:
                return formula
        
        # Look for single numbers like (2) vs (3)
        match1 = re.search(r'\((\d+)\)', col1)
        match2 = re.search(r'\((\d+)\)', col2)
        if match1 and match2 and match1.group(1) != match2.group(1):
            return match1.group(1)
        
        return ""


# Global LLM architect instance
llm_architect = LLMArchitect()
