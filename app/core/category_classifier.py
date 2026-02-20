from typing import Dict
import pandas as pd
import json
from openai import OpenAI

from app.config import settings
from app.core.logger import logger


class CategoryClassifier:
    """
    Classifies a DataFrame as Category 1 (standard) or Category 2 (hierarchical,
    with section headers, subtotals, footers mixed with data rows).
    """

    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model

    def classify(self, df: pd.DataFrame, analysis: Dict) -> Dict:
        """
        Classify a DataFrame as Category 1 or Category 2.

        Args:
            df: Raw DataFrame (after header detection + loading)
            analysis: LLM analysis dict from llm_architect.analyze_file_structure()

        Returns:
            {
                "category": 1 | 2,
                "confidence": float (0.0-1.0),
                "reasoning": str
            }
        """
        try:
            # Compact summary: shape, columns, first 5 + last 5 rows
            shape = df.shape
            columns = df.columns.tolist()
            if isinstance(df.columns, pd.MultiIndex):
                columns = [str(c) for c in df.columns]

            first_5 = df.head(5).to_dict(orient="records")
            last_5 = df.tail(5).to_dict(orient="records")

            # Check for mostly-empty columns (signals section headers)
            empty_pct = (df.isna().sum() / len(df)).to_dict()
            mostly_empty = [c for c, pct in empty_pct.items() if pct > 0.5]

            prompt = f"""You are classifying a data table into one of two categories.

Category 1: Standard table. All rows are data rows. Column headers may be complex (multi-level), but there are no section header rows, subtotal rows, or footer rows mixed with the data.

Category 2: Lightly structured table. The table contains NON-DATA ROWS mixed with data rows, such as:
- Section headers (rows that label a group/section but have no or mostly empty numeric values)
- Subtotal/total rows (aggregations like "Sub Total", "Total", "All India Total")
- Footer rows (source citations, disclaimers, notes like "Source: Ministry of X", "Total may not tally due to rounding")

Table summary:
- Shape: {shape[0]} rows × {shape[1]} columns
- Column names: {columns}
- Columns with >50% empty values (often indicate section headers): {mostly_empty if mostly_empty else "none"}

First 5 rows (sample):
{json.dumps(first_5, indent=2, default=str)}

Last 5 rows (sample):
{json.dumps(last_5, indent=2, default=str)}

Does this table contain non-data rows (section headers, subtotals, footers) mixed with data rows?
- If YES → Category 2
- If NO (all rows are data) → Category 1

Respond in JSON format only:
{{
    "category": 1 or 2,
    "confidence": 0.0 to 1.0,
    "reasoning": "brief explanation"
}}"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a data structure expert. Classify tables as Category 1 (all data rows) or Category 2 (mixed section headers, subtotals, footers). Respond only with valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )

            result = json.loads(response.choices[0].message.content)
            category = int(result.get("category", 1))
            confidence = float(result.get("confidence", 0.0))
            reasoning = result.get("reasoning", "")

            # Enforce valid category
            if category not in (1, 2):
                category = 1
            if confidence < 0.6:
                category = 1
                reasoning = f"Low confidence ({confidence}), defaulting to Cat1. {reasoning}"

            logger.info(f"CategoryClassifier: category={category}, confidence={confidence}")
            return {"category": category, "confidence": confidence, "reasoning": reasoning}

        except Exception as e:
            logger.error(f"CategoryClassifier failed: {e}", exc_info=True)
            return {
                "category": 1,
                "confidence": 0.0,
                "reasoning": f"LLM unavailable, defaulting to Cat1: {str(e)}",
            }


category_classifier = CategoryClassifier()
