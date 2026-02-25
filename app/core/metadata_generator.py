from datetime import datetime
from typing import Dict, Optional
from app.core.logger import logger
from app.core.llm_architect import llm_architect
import pandas as pd
import json


class MetadataGenerator:
    """
    Generates metadata for ingested tables.
    Combines LLM-generated fields with user-provided inputs.
    """
    
    # Valid data domains
    VALID_DOMAINS = [
        "CPI", "GDP", "IIP", "MSME", "GST", 
        "agriculture_and_rural", "social_migration_and_households",
        "enterprise_surveys", "worker_surveys", "finance_and_industry"
    ]
    
    # Valid major domains for operational_metadata
    VALID_MAJOR_DOMAINS = [
        "Agriculture, Forestry and Rural Development", "Central Bank Assets",
        "Chemical, Mining and Natural Resources", "Commerce, Finance, Banking and Insurance",
        "Credit Statistics", "Demography", "Derivatives Statistics", "Digital payments",
        "Economy and Financial Indicators", "E-Governance", "Enterprise Establishment Surveys",
        "Environment", "Environment and Trade", "EPFO", "Exchange Rates", "FDI",
        "Financial Securities", "GDP", "Global Liquidity Indicators", "GST",
        "Healthcare", "Household Consumption Expenditure", "IIP", "Industry productivity",
        "Insurance", "International Banking", "Labour", "Logistics and Mobility",
        "Ministry of Corporate Affairs", "MSME", "Mutual Funds", "Prices & Inflation",
        "Rainfall", "Real Estate", "Renewable Resources", "Research and Development",
        "State Economic Survey", "Toll", "Trade", "Trade Policy Uncertainty",
        "Traffic", "Transportation", "UNIDO", "Youth Development"
    ]
    
    def classify_domain(self, df: pd.DataFrame, table_name: str, analysis: Dict) -> str:
        """
        Use LLM to classify the data domain from predefined list.
        
        Args:
            df: DataFrame being processed
            table_name: Generated table name
            analysis: Previous analysis results
            
        Returns:
            Classified domain from VALID_DOMAINS
        """
        logger.info("Classifying data domain with LLM")
        
        sample_data = df.head(5).to_dict(orient='records')
        
        prompt = f"""Classify this dataset into ONE of the following data domains:

Valid Domains:
{', '.join(self.VALID_DOMAINS)}

Dataset Information:
- Table Name: {table_name}
- Columns: {df.columns.tolist()}
- Sample Data: {json.dumps(sample_data, indent=2, default=str)}
- Previous Analysis: {analysis.get('domain', 'unknown')}

Based on the table name, columns, and sample data, select the MOST APPROPRIATE domain from the list above.
The selected domain abbreviation will be used directly in the table name (e.g. IIP, CPI, GDP), so choose the most specific match.

Respond in JSON format:
{{
    "domain": "selected_domain",
    "reasoning": "brief explanation"
}}"""
        
        try:
            response = llm_architect.client.chat.completions.create(
                model=llm_architect.model,
                messages=[
                    {"role": "system", "content": "You are a data classification expert. Classify datasets into predefined domains."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2
            )
            
            result = json.loads(response.choices[0].message.content)
            domain = result.get("domain", "IIP")
            
            # Validate domain
            if domain not in self.VALID_DOMAINS:
                logger.warning(f"Invalid domain '{domain}', defaulting to 'IIP'")
                domain = "IIP"
            
            logger.info(f"Classified domain: {domain} ({result.get('reasoning', '')})")
            return domain
            
        except Exception as e:
            logger.error(f"Error classifying domain: {str(e)}")
            return "IIP"  # Default fallback
    
    def generate_description(self, df: pd.DataFrame, table_name: str) -> str:
        """
        Generate 1-2 line description of the table and data.
        
        Args:
            df: DataFrame being processed
            table_name: Table name
            
        Returns:
            Brief description
        """
        logger.info("Generating table description with LLM")
        
        sample_data = df.head(5).to_dict(orient='records')
        
        prompt = f"""Generate a concise 1-2 sentence description of this dataset.

Table Name: {table_name}
Columns: {df.columns.tolist()}
Sample Data: {json.dumps(sample_data, indent=2, default=str)}
Total Rows: {len(df)}

The description should:
- Be 1-2 sentences maximum
- Describe what data the table contains
- Mention the geographic scope, time granularity, and data grain where evident (e.g. "India monthly data at sub-category level")
- Mention the time period if evident
- Be clear and professional

Respond in JSON format:
{{
    "description": "your description here"
}}"""
        
        try:
            response = llm_architect.client.chat.completions.create(
                model=llm_architect.model,
                messages=[
                    {"role": "system", "content": "You are a data documentation expert. Write concise, clear descriptions."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            result = json.loads(response.choices[0].message.content)
            description = result.get("description", f"Data table: {table_name}")
            
            logger.info(f"Generated description: {description}")
            return description
            
        except Exception as e:
            logger.error(f"Error generating description: {str(e)}")
            return f"Data table containing {len(df)} rows"
    
    def detect_period_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Detect which column contains period/time information.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Column name containing period data, or None
        """
        logger.info("Detecting period column")
        
        # Common period column name patterns
        period_patterns = ['year', 'month', 'quarter', 'date', 'period', 'time', 'fiscal']
        
        # Check column names
        for col in df.columns:
            col_lower = str(col).lower()
            if any(pattern in col_lower for pattern in period_patterns):
                logger.info(f"Detected period column: {col}")
                return col
        
        # If no obvious match, use LLM
        try:
            sample_data = df.head(10).to_dict(orient='records')
            
            prompt = f"""Identify which column contains period/time information (year, month, quarter, date, etc.).

Columns: {df.columns.tolist()}
Sample Data: {json.dumps(sample_data, indent=2, default=str)}

Respond in JSON format:
{{
    "period_column": "column_name or null if none found",
    "reasoning": "brief explanation"
}}"""
            
            response = llm_architect.client.chat.completions.create(
                model=llm_architect.model,
                messages=[
                    {"role": "system", "content": "You are a data analysis expert. Identify time/period columns."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2
            )
            
            result = json.loads(response.choices[0].message.content)
            period_col = result.get("period_column")
            
            if period_col and period_col in df.columns:
                logger.info(f"LLM detected period column: {period_col}")
                return period_col
            
        except Exception as e:
            logger.error(f"Error detecting period column: {str(e)}")
        
        return None
    
    def generate_period_sql(self, table_name: str, period_col: str) -> tuple[str, str]:
        """
        Generate SQL queries for min and max period.
        
        Args:
            table_name: Name of the table
            period_col: Name of the period column
            
        Returns:
            Tuple of (min_period_sql, max_period_sql)
        """
        logger.info(f"Generating period SQL for column: {period_col}")
        
        # Simple queries for single period column
        min_sql = f'SELECT MIN("{period_col}") AS min_period FROM "{table_name}";'
        max_sql = f'SELECT MAX("{period_col}") AS max_period FROM "{table_name}";'
        
        logger.info(f"Generated min SQL: {min_sql}")
        logger.info(f"Generated max SQL: {max_sql}")
        
        return min_sql, max_sql
    
    def classify_major_domain(self, df: pd.DataFrame, table_name: str, analysis: Dict) -> str:
        """
        Use LLM to classify the major domain from predefined list for operational_metadata.
        
        Args:
            df: DataFrame being processed
            table_name: Generated table name
            analysis: Previous analysis results
            
        Returns:
            Classified major domain from VALID_MAJOR_DOMAINS
        """
        logger.info("Classifying major domain with LLM")
        
        sample_data = df.head(5).to_dict(orient='records')
        
        prompt = f"""Classify this dataset into ONE of the following major domains:

Valid Major Domains:
{chr(10).join(['- ' + d for d in self.VALID_MAJOR_DOMAINS])}

Dataset Information:
- Table Name: {table_name}
- Columns: {df.columns.tolist()}
- Sample Data: {json.dumps(sample_data, indent=2, default=str)}

Based on the table name, columns, and sample data, select the MOST APPROPRIATE major domain from the list above.
The domain must be EXACTLY as written in the list (including capitalization and punctuation).

Respond in JSON format:
{{
    "major_domain": "selected_domain_exactly_as_listed",
    "reasoning": "brief explanation"
}}"""
        
        try:
            response = llm_architect.client.chat.completions.create(
                model=llm_architect.model,
                messages=[
                    {"role": "system", "content": "You are a data classification expert. Classify datasets into predefined major domains."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2
            )
            
            result = json.loads(response.choices[0].message.content)
            major_domain = result.get("major_domain", "Economy and Financial Indicators")
            
            # Validate major domain
            if major_domain not in self.VALID_MAJOR_DOMAINS:
                logger.warning(f"Invalid major domain '{major_domain}', defaulting to 'Economy and Financial Indicators'")
                major_domain = "Economy and Financial Indicators"
            
            logger.info(f"Classified major domain: {major_domain} ({result.get('reasoning', '')})")
            return major_domain
            
        except Exception as e:
            logger.error(f"Error classifying major domain: {str(e)}")
            return "Economy and Financial Indicators"  # Default fallback
    
    def classify_sub_domain(self, df: pd.DataFrame, table_name: str, major_domain: str) -> str:
        """
        Use LLM to determine sub-domain based on the selected major domain.
        
        Args:
            df: DataFrame being processed
            table_name: Generated table name
            major_domain: Previously classified major domain
            
        Returns:
            Sub-domain classification
        """
        logger.info(f"Classifying sub-domain for major domain: {major_domain}")
        
        sample_data = df.head(5).to_dict(orient='records')
        
        prompt = f"""Based on the major domain classification, determine an appropriate sub-domain for this dataset.

Major Domain: {major_domain}

Dataset Information:
- Table Name: {table_name}
- Columns: {df.columns.tolist()}
- Sample Data: {json.dumps(sample_data, indent=2, default=str)}

Provide a specific sub-domain that describes the particular aspect or category within the major domain.
The sub-domain should be concise (2-4 words) and descriptive.

Examples:
- Major Domain: "Digital payments" → Sub-domain: "UPI Transaction Statistics"
- Major Domain: "Trade" → Sub-domain: "Export-Import Indices"
- Major Domain: "IIP" → Sub-domain: "Manufacturing Sector"

Respond in JSON format:
{{
    "sub_domain": "your sub-domain classification",
    "reasoning": "brief explanation"
}}"""
        
        try:
            response = llm_architect.client.chat.completions.create(
                model=llm_architect.model,
                messages=[
                    {"role": "system", "content": "You are a data classification expert. Determine appropriate sub-domains based on major domain classifications."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            result = json.loads(response.choices[0].message.content)
            sub_domain = result.get("sub_domain", "General Statistics")
            
            logger.info(f"Classified sub-domain: {sub_domain} ({result.get('reasoning', '')})")
            return sub_domain
            
        except Exception as e:
            logger.error(f"Error classifying sub-domain: {str(e)}")
            return "General Statistics"
    
    def get_period_values(self, df: pd.DataFrame, period_col: str) -> tuple[Optional[str], Optional[str]]:
        """
        Get first and last available values from the period column.
        
        Args:
            df: DataFrame being processed
            period_col: Name of the period column
            
        Returns:
            Tuple of (first_available_value, last_available_value)
        """
        if not period_col or period_col not in df.columns:
            logger.warning(f"Period column '{period_col}' not found in DataFrame")
            return None, None
        
        try:
            # Get non-null values from period column
            period_values = df[period_col].dropna()
            
            if len(period_values) == 0:
                logger.warning(f"No non-null values in period column '{period_col}'")
                return None, None
            
            # Try to sort if possible (for dates/numbers)
            try:
                period_values_sorted = period_values.sort_values()
                first_value = str(period_values_sorted.iloc[0])
                last_value = str(period_values_sorted.iloc[-1])
            except:
                # If sorting fails, just use min/max as strings
                first_value = str(period_values.min())
                last_value = str(period_values.max())
            
            logger.info(f"Period range: {first_value} to {last_value}")
            return first_value, last_value
            
        except Exception as e:
            logger.error(f"Error getting period values: {str(e)}")
            return None, None


# Global metadata generator instance
metadata_generator = MetadataGenerator()
