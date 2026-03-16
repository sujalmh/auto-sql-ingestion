"""
Excel structure analyzer using openpyxl.

Detects title rows, header rows, column-numbering rows, merged cells,
footer/footnote rows, and data boundaries in complex Excel files
BEFORE pandas loads them.  This avoids the common pitfall of pandas
flattening merged cells into NaN-filled rows that confuse downstream
header detection and LLM analysis.
"""

import re
from typing import Dict, List, Optional, Tuple

from openpyxl import load_workbook
from openpyxl.utils import get_column_letter

from app.core.logger import logger


class ExcelStructure:
    """Result of Excel structure analysis."""

    def __init__(
        self,
        *,
        skip_rows: int = 0,
        header_rows: List[int],
        num_header_rows: int = 1,
        col_number_row: Optional[int] = None,
        data_start_row: int,
        data_end_row: int,
        footer_start_row: Optional[int] = None,
        min_data_col: int = 1,
        max_data_col: int = 1,
        unit_hint: Optional[str] = None,
        title_hint: Optional[str] = None,
        merged_header_values: Optional[List[List[Optional[str]]]] = None,
    ):
        self.skip_rows = skip_rows
        self.header_rows = header_rows            # 1-indexed Excel row numbers
        self.num_header_rows = num_header_rows
        self.col_number_row = col_number_row      # 1-indexed, or None
        self.data_start_row = data_start_row      # 1-indexed, first actual data row
        self.data_end_row = data_end_row          # 1-indexed, last actual data row
        self.footer_start_row = footer_start_row
        self.min_data_col = min_data_col          # 1-indexed
        self.max_data_col = max_data_col          # 1-indexed
        self.unit_hint = unit_hint                # e.g. "₹ Crores"
        self.title_hint = title_hint              # extracted title text
        self.merged_header_values = merged_header_values  # resolved header grid

    def to_dict(self) -> Dict:
        return {
            "skip_rows": self.skip_rows,
            "header_rows": self.header_rows,
            "num_header_rows": self.num_header_rows,
            "col_number_row": self.col_number_row,
            "data_start_row": self.data_start_row,
            "data_end_row": self.data_end_row,
            "footer_start_row": self.footer_start_row,
            "min_data_col": self.min_data_col,
            "max_data_col": self.max_data_col,
            "unit_hint": self.unit_hint,
            "title_hint": self.title_hint,
        }


# ── Patterns ──────────────────────────────────────────────────────────

_FOOTER_PATTERNS = re.compile(
    r"^\s*(notes?\s*:|source\s*:|"
    r"\*|@|#|†|‡|§|"
    r"\d+\.\s+[A-Z]|"                       # "1. Data for ..."
    r"p\s*[-–:]\s*provisional|"
    r"re\s*[-–:]\s*revised|"
    r"be\s*[-–:]\s*budget)",
    re.IGNORECASE,
)

_UNIT_PATTERNS = re.compile(
    r"\(?\s*₹\s*\w+\s*\)?|"
    r"\(\s*(?:Rs|INR|rupee|crore|lakh|percent|ratio)\b[^)]*\)",
    re.IGNORECASE,
)

_COL_NUMBER_RE = re.compile(r"^\s*\d+\s*$")


class ExcelAnalyzer:
    """Analyze Excel (.xlsx) file structure before pandas loading."""

    def analyze(self, file_path: str, sheet_name: Optional[str] = None) -> ExcelStructure:
        """
        Analyze the Excel file and return structural metadata.

        The workbook is opened in data_only mode (formulas resolved to values)
        and NOT in read_only mode so we can access merged_cells info.
        """
        logger.info(f"ExcelAnalyzer: analysing {file_path}")

        wb = load_workbook(file_path, read_only=False, data_only=True)
        ws = wb[sheet_name] if sheet_name else wb.active
        if ws is None:
            wb.close()
            raise ValueError(f"No active sheet found in {file_path}")
        max_row = ws.max_row or 1
        max_col = ws.max_column or 1

        # ── 1. Build a merged-cell lookup ──────────────────────────
        merged_map = self._build_merged_map(ws)

        # ── 2. Read all cell values (with merged-cell fill) ───────
        grid = self._read_grid(ws, max_row, max_col, merged_map)

        # ── 3. Detect title / metadata rows at the top ────────────
        title_hint, unit_hint, first_content_row = self._detect_title_rows(
            grid, max_row, max_col
        )

        # ── 4. Detect header rows ─────────────────────────────────
        header_rows, col_number_row = self._detect_headers(
            grid, first_content_row, max_row, max_col
        )

        # ── 5. Detect data column range ───────────────────────────
        min_data_col, max_data_col = self._detect_col_range(
            grid, header_rows, max_col
        )

        # ── 6. Detect footer rows at the bottom ──────────────────
        data_start_row = (col_number_row or header_rows[-1]) + 1 if header_rows else first_content_row
        footer_start_row, data_end_row = self._detect_footer(
            grid, data_start_row, max_row, max_col
        )

        # ── 7. Resolve merged header values ───────────────────────
        merged_header_values = self._resolve_header_values(
            grid, header_rows, min_data_col, max_data_col
        )

        # Convert header_rows to 0-indexed "skip_rows" count for pandas
        # skip_rows = number of rows before the first header row
        skip_rows = header_rows[0] - 1 if header_rows else 0

        wb.close()

        structure = ExcelStructure(
            skip_rows=skip_rows,
            header_rows=header_rows,
            num_header_rows=len(header_rows),
            col_number_row=col_number_row,
            data_start_row=data_start_row,
            data_end_row=data_end_row,
            footer_start_row=footer_start_row,
            min_data_col=min_data_col,
            max_data_col=max_data_col,
            unit_hint=unit_hint,
            title_hint=title_hint,
            merged_header_values=merged_header_values,
        )

        logger.info(f"ExcelAnalyzer result: {structure.to_dict()}")
        return structure

    # ── Private helpers ───────────────────────────────────────────────

    def _build_merged_map(self, ws) -> Dict[Tuple[int, int], Tuple[int, int]]:
        """
        Build a dict mapping every cell inside a merged range to the
        top-left anchor cell of that range.
        Key: (row, col) 1-indexed  →  Value: (anchor_row, anchor_col) 1-indexed
        """
        merged_map: Dict[Tuple[int, int], Tuple[int, int]] = {}
        for merge_range in ws.merged_cells.ranges:
            min_r, min_c = merge_range.min_row, merge_range.min_col
            for r in range(merge_range.min_row, merge_range.max_row + 1):
                for c in range(merge_range.min_col, merge_range.max_col + 1):
                    if (r, c) != (min_r, min_c):
                        merged_map[(r, c)] = (min_r, min_c)
        return merged_map

    def _read_grid(
        self, ws, max_row: int, max_col: int,
        merged_map: Dict[Tuple[int, int], Tuple[int, int]],
    ) -> List[List]:
        """
        Read the worksheet into a 2D list (1-indexed via grid[row][col]).
        Merged cells are filled with the anchor cell's value.
        grid[0] is unused (placeholder) so that grid[1][1] = cell A1.
        """
        # Pre-read all cell values
        raw: Dict[Tuple[int, int], object] = {}
        for row in ws.iter_rows(min_row=1, max_row=max_row, max_col=max_col, values_only=False):
            for cell in row:
                if cell.value is not None:
                    raw[(cell.row, cell.column)] = cell.value

        # Build grid with merged-cell propagation
        grid: List[List] = [[None] * (max_col + 1)]  # row 0 placeholder
        for r in range(1, max_row + 1):
            row_vals: List = [None]  # col 0 placeholder
            for c in range(1, max_col + 1):
                if (r, c) in merged_map:
                    anchor = merged_map[(r, c)]
                    row_vals.append(raw.get(anchor))
                else:
                    row_vals.append(raw.get((r, c)))
            grid.append(row_vals)
        return grid

    def _detect_title_rows(
        self, grid: List[List], max_row: int, max_col: int,
    ) -> Tuple[Optional[str], Optional[str], int]:
        """
        Detect title and unit rows at the top of the sheet.
        Returns (title_text, unit_text, first_content_row).

        Title rows: rows with text in only 1-2 columns, or a single merged text
        spanning many columns.  They appear before the real headers.
        """
        title_hint = None
        unit_hint = None
        first_content_row = 1

        for r in range(1, min(max_row + 1, 12)):  # scan first 11 rows
            row = grid[r]
            non_null = [(c, v) for c, v in enumerate(row[1:], 1) if v is not None]

            if len(non_null) == 0:
                # empty row — could be separator between title and headers
                continue

            # Check if this is a unit hint row like "(₹ Crores)"
            if len(non_null) == 1:
                val_str = str(non_null[0][1]).strip()
                if _UNIT_PATTERNS.search(val_str):
                    unit_hint = val_str.strip("() ")
                    continue

            # Check if all non-null values are the same (merged title)
            unique_vals = set(str(v) for _, v in non_null)
            if len(unique_vals) == 1 and len(non_null) >= 2:
                # Merged title row
                if title_hint is None:
                    title_hint = str(non_null[0][1]).strip()
                continue

            # Single value in one cell — likely a title or unit
            if len(non_null) == 1:
                val_str = str(non_null[0][1]).strip()
                # Short single-cell text before headers = likely title/unit
                if len(val_str) < 200 and not _COL_NUMBER_RE.match(val_str):
                    if title_hint is None and not _UNIT_PATTERNS.search(val_str):
                        title_hint = val_str
                    continue

            # Multiple non-null values in different columns → likely a header or data row
            # Check if at least 3 columns have values (characteristic of headers)
            if len(non_null) >= 3:
                first_content_row = r
                break

        logger.info(
            f"ExcelAnalyzer: title='{title_hint}', unit='{unit_hint}', "
            f"first_content_row={first_content_row}"
        )
        return title_hint, unit_hint, first_content_row

    def _detect_headers(
        self, grid: List[List], first_content_row: int, max_row: int, max_col: int,
    ) -> Tuple[List[int], Optional[int]]:
        """
        Starting from first_content_row, detect header and column-numbering rows.
        Returns (header_row_numbers, col_number_row_or_None).

        Header rows have many string values across columns.
        Column-numbering rows have sequential integers (1, 2, 3, ...).
        """
        header_rows: List[int] = []
        col_number_row: Optional[int] = None

        for r in range(first_content_row, min(max_row + 1, first_content_row + 10)):
            row = grid[r]
            non_null = [(c, v) for c, v in enumerate(row[1:], 1) if v is not None]
            if len(non_null) == 0:
                continue

            # Check if this is a column-numbering row (1, 2, 3, ...)
            if self._is_column_number_row(non_null):
                col_number_row = r
                continue

            # Check if this looks like a header row
            str_count = sum(1 for _, v in non_null if isinstance(v, str) and not _COL_NUMBER_RE.match(str(v).strip()))
            num_count = sum(1 for _, v in non_null if isinstance(v, (int, float)) and not isinstance(v, bool))

            # A header row has predominantly string values
            if str_count >= 2 and str_count >= num_count:
                header_rows.append(r)
            elif num_count > str_count and len(header_rows) > 0:
                # We've hit data rows; stop
                break

        if not header_rows:
            header_rows = [first_content_row]

        logger.info(
            f"ExcelAnalyzer: header_rows={header_rows}, "
            f"col_number_row={col_number_row}"
        )
        return header_rows, col_number_row

    def _is_column_number_row(self, non_null: List[Tuple[int, object]]) -> bool:
        """Check if a row contains sequential column numbers like 1, 2, 3, ..."""
        if len(non_null) < 3:
            return False

        numbers = []
        for _, v in non_null:
            try:
                n = int(float(str(v).strip()))
                numbers.append(n)
            except (ValueError, TypeError):
                return False

        if not numbers:
            return False

        # Check if they're roughly sequential starting from 1
        # Allow small gaps (some columns might be empty)
        return numbers[0] == 1 and numbers[-1] >= len(numbers) * 0.5

    def _detect_col_range(
        self, grid: List[List], header_rows: List[int], max_col: int,
    ) -> Tuple[int, int]:
        """Detect the range of columns that contain actual data."""
        min_col = max_col
        max_col_found = 1

        for r in header_rows:
            row = grid[r]
            for c in range(1, len(row)):
                if row[c] is not None:
                    min_col = min(min_col, c)
                    max_col_found = max(max_col_found, c)

        return min_col, max_col_found

    def _detect_footer(
        self, grid: List[List], data_start: int, max_row: int, max_col: int,
    ) -> Tuple[Optional[int], int]:
        """
        Scan from the bottom up to find footer/notes rows.
        Returns (footer_start_row_or_None, last_data_row).
        """
        footer_start = None
        last_data_row = max_row

        for r in range(max_row, max(data_start - 1, 0), -1):
            row = grid[r]
            non_null = [(c, v) for c, v in enumerate(row[1:], 1) if v is not None]

            if len(non_null) == 0:
                # Empty row — could be separator between data and footer
                last_data_row = r - 1
                continue

            # Check if this row matches footer patterns
            first_val = str(non_null[0][1]).strip()
            if _FOOTER_PATTERNS.match(first_val):
                footer_start = r
                last_data_row = r - 1
                continue

            # Check if only 1-2 columns have values and it's text-heavy
            # (characteristic of footnotes in last rows)
            if len(non_null) <= 2 and all(isinstance(v, str) for _, v in non_null):
                text = " ".join(str(v) for _, v in non_null)
                if len(text) > 50:  # Long text in 1-2 columns = likely footnote
                    footer_start = r
                    last_data_row = r - 1
                    continue

            # This row has real data — stop scanning
            break

        # Adjust: if last_data_row still points at an empty row, move up
        while last_data_row > data_start and all(
            v is None for v in grid[last_data_row][1:]
        ):
            last_data_row -= 1

        logger.info(
            f"ExcelAnalyzer: footer_start={footer_start}, "
            f"last_data_row={last_data_row}"
        )
        return footer_start, last_data_row

    def _resolve_header_values(
        self, grid: List[List], header_rows: List[int],
        min_col: int, max_col: int,
    ) -> List[List[Optional[str]]]:
        """
        Build a resolved header grid where merged cells are properly filled.
        Returns a list of lists, one per header row, with string values for
        each column in the [min_col, max_col] range.
        """
        if not header_rows:
            return []

        result = []
        for r in header_rows:
            row_vals = []
            for c in range(min_col, max_col + 1):
                v = grid[r][c] if c < len(grid[r]) else None
                row_vals.append(str(v).strip() if v is not None else None)
            result.append(row_vals)
        return result


# Module-level singleton
excel_analyzer = ExcelAnalyzer()
