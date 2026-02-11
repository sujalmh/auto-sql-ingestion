from openai import OpenAI
import pandas as pd
import json
from typing import Dict, List, Tuple
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
1. Identify if there are date/timestamp values in column headers that should be transformed into rows
2. Detect if headers span multiple levels and need merging. This includes:
   - Pandas MultiIndex columns
   - First few rows containing header information (e.g., "Unit value index" spanning multiple columns)
   - Look for patterns where first rows have duplicate values or many NaN values
3. Identify the grain/dimensionality of the data (e.g., State-wise, Category-wise, Monthly)
4. Suggest preprocessing strategy

IMPORTANT: Check if the first 1-3 rows look like headers rather than data. Headers typically have:
- Duplicate values across columns (indicating grouped headers)
- Many NaN/empty values
- Text values when rest of data is numeric

Respond in JSON format:
{{
    "has_date_headers": true/false,
    "date_columns": ["list of column names with dates"],
    "needs_header_merge": true/false,
    "header_merge_reason": "explanation if needs_header_merge is true",
    "data_grain": "description of data granularity",
    "domain": "data domain (e.g., IIP, GDP, Trade, etc.)",
    "preprocessing_strategy": "brief description of recommended preprocessing"
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
                "preprocessing_strategy": "standard cleaning"
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
Naming Convention: auto_<domain>_<Geo>_<Time>_<grain-Dimension>

Examples:
- auto_IIP_India_Mth_SCtg (IIP India Monthly SubCategory)
- auto_IIP_State_Mth_Catg (IIP Statewise Monthly Category)
- auto_IIP_State_Mth_Sector (IIP Statewise Monthly Sector)

Guidelines:
- domain: Data domain keep it in full form only unless there exists its short form eg : GDP, CPI, etc.
- Geo: Geographic level (India, State, District, etc.)
- Time: Time granularity (Mth=Monthly, Qtr=Quarterly, Yr=Yearly, etc.)
- grain-Dimension: Data grain/dimension (Catg=Category, SCtg=SubCategory, Sector, etc.)
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

Generate an appropriate table name following the convention. The name should be:
- All lowercase
- Use underscores to separate parts
- Be descriptive but concise

Respond in JSON format:
{{
    "table_name": "auto_<domain>_<geo>_<time>_<grain-Dimension>",
    "reasoning": "brief explanation of the naming choice"
}}"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a database naming expert. Generate table names following strict naming conventions."},
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
            prompt = f"""Analyze this Excel/CSV data preview and count ONLY the header rows (column labels), NOT data rows.

Data Preview:
{markdown}

CRITICAL RULES:
1. Header rows contain DESCRIPTIVE LABELS (e.g., "Year", "State", "GDP", "Category")
2. Data rows contain ACTUAL VALUES (e.g., "2023", "Maharashtra", "15000", specific numbers/text)
3. If the FIRST row has descriptive labels and SECOND row has actual data → return 1
4. If FIRST TWO rows both have descriptive labels (e.g., main category + subcategory) → return 2
5. If FIRST THREE rows all have descriptive labels → return 3

COMMON MISTAKE TO AVOID:
- DO NOT count a data row as a header just because it's in the first few rows
- Look for patterns: headers are typically short descriptive text, data rows have varied values

Examples:
- Row 1: "Year | State | Value" → 1 header row
- Row 1: "Economic Indicators" Row 2: "Year | State | Value" → 2 header rows
- Row 1: "2023 | Delhi | 5000" → 1 header row (the actual column names are likely row 0, which is already parsed)

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
            prompt = f"""Shorten this table name to max 40 characters.

Original: {table_name}

Rules:
- Keep domain (monetary, trade, gdp)
- Keep geography (india)
- Keep time grain (yr, qtr, mth)
- Use underscores, lowercase
- Economics context

Return ONLY the shortened name."""
            
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
    
    def clean_column_names(self, columns: list) -> dict:
        """Clean messy column names ensuring uniqueness."""
        logger.info(f"Cleaning {len(columns)} column names")
        try:
            import json
            prompt = f"""You are cleaning database column names. CRITICAL: All output names MUST be unique.

INPUT COLUMNS:
{json.dumps(columns, indent=2)}

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

Return JSON mapping each input to its unique output name:
{{"input_column_name": "unique_output_name", ...}}"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1  # Lower temperature for more consistent output
            )
            mapping = json.loads(response.choices[0].message.content)
            
            # Validate uniqueness of cleaned names
            cleaned_values = list(mapping.values())
            duplicates = [v for v in set(cleaned_values) if cleaned_values.count(v) > 1]
            
            if duplicates:
                logger.warning(f"LLM returned {len(duplicates)} duplicate names: {duplicates}")
                logger.warning("Applying post-processing to fix duplicates")
                mapping = self._ensure_unique_column_names(columns, mapping)
            
            logger.info(f"Cleaned {len(mapping)} column names (all unique)")
            return mapping
        except Exception as e:
            logger.error(f"Error cleaning names: {str(e)}")
            return {col: col for col in columns}
    
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
