"""Shared column name normalization for similarity and incremental load matching."""
import re

# Months pattern used in multiple regexes
_MONTHS = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"


def normalize_column_for_similarity(col: str) -> str:
    """Normalize column name so period/date differences don't separate matches.

    Strips the following period-varying suffixes so that columns like
    'no_of_schemes_as_on_dec_31_2025' and 'no_of_schemes_as_on_jan_31_2026'
    both normalize to 'no_of_schemes_as_on'.

    Handled patterns (order matters — most specific first):
      1. Month-day-year:  _jan_31_2024, _december_1_2025
      2. ISO dates:       _2024-01-31, _2024_01_31
      3. Month-year:      _jan_2024, _december_2025, _dec-2024
      4. FY prefix:       _fy2024_25, _fy_2024, _fy24_25
      5. Quarter:         _q1_2024, _q2_2024_25
      6. Fiscal year:     _2023_24, _2024_25
      7. Bare year (end): _2024, _2025  (only 1900-2099 at end of string)
    """
    if not col:
        return col
    s = col.lower()
    # 1. Month-day-year: _jan_31_2024, _december_1_2025
    s = re.sub(rf"_?{_MONTHS}[-_]?\d{{1,2}}[-_]?\d{{4}}", "", s)
    # 2. ISO/underscore dates: _2024-01-01, _2024_01_01
    s = re.sub(r"_\d{4}[-_]\d{2}[-_]\d{2}", "", s)
    # 3. Month-year (no day): _jan_2024, _december_2025, _dec-2024
    s = re.sub(rf"_?{_MONTHS}[-_]?(?:19|20)\d{{2}}", "", s)
    # 4. FY prefix: _fy2024_25, _fy_2024_25, _fy24_25, _fy_2024
    s = re.sub(r"_fy[-_]?(?:19|20)?\d{2}(?:[-_]\d{2})?", "", s)
    # 5. Quarter: _q1_2024, _q2_2024_25
    s = re.sub(r"_q[1-4][-_]?(?:19|20)?\d{2}(?:[-_]\d{2})?", "", s)
    # 6. Fiscal year: _2023_24, _2024_25 (4-digit + underscore + 2-digit)
    s = re.sub(r"_(?:19|20)\d{2}[-_]\d{2}\b", "", s)
    # 7. Bare year at end: _2024, _2025 (only realistic years)
    s = re.sub(r"_(?:19|20)\d{2}$", "", s)
    # Clean up leftover separators
    s = re.sub(r"_+", "_", s).strip("_")
    return s or col.lower()
