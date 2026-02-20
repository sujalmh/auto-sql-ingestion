"""Shared column name normalization for similarity and incremental load matching."""
import re


def normalize_column_for_similarity(col: str) -> str:
    """Normalize column name so period/date differences don't separate matches.
    E.g. 'no_of_schemes_as_on_dec_31_2025' and 'no_of_schemes_as_on_jan_31_2026' -> same base."""
    if not col:
        return col
    s = col.lower()
    s = re.sub(r"_?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)_?\d{1,2}_?\d{4}", "", s)
    s = re.sub(r"_\d{4}[-_]\d{2}[-_]\d{2}", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or col.lower()
