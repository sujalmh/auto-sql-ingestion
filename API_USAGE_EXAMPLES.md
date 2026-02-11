# API Usage Examples - Dual Metadata Tracking System

## 1. Upload File
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@your_data.xlsx"
```

**Response:**
```json
{
  "job_id": "abc123-def456-ghi789",
  "message": "File uploaded successfully. Processing started.",
  "status": "preprocessing"
}
```

---

## 2. Check Status (Get Preview with LLM Metadata)
```bash
curl "http://localhost:8000/status/abc123-def456-ghi789"
```

**Response:**
```json
{
  "job_id": "abc123-def456-ghi789",
  "status": "awaiting_approval",
  "created_at": "2026-01-09T16:00:00",
  "updated_at": "2026-01-09T16:00:30",
  "preview": {
    "proposed_table_name": "trade_india_yr_index",
    "columns": [...],
    "sample_rows": [...],
    "total_rows": 1572,
    "preprocessing_summary": "Merged headers, transformed date columns...",
    "llm_metadata": {
      "suggested_domain": "Trade and Commerce",
      "description": "Trade statistics for India including exports and imports",
      "period_column": "year"
    }
  }
}
```

---

## 3. Approve with Metadata (NEW - Individual Fields!)

### Option A: Using cURL
```bash
curl -X POST "http://localhost:8000/approve/abc123-def456-ghi789" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "source=Ministry of Commerce and Industry" \
  -d "source_url=https://commerce.gov.in/trade-statistics" \
  -d "released_on=2025-08-11T00:00:00" \
  -d "updated_on=2025-08-11T00:00:00" \
  -d "business_metadata=Monthly trade data for policy analysis" \
  -d "table_name=trade_india_yr_index"
```

### Option B: Using Python requests
```python
import requests

response = requests.post(
    "http://localhost:8000/approve/abc123-def456-ghi789",
    data={
        "source": "Ministry of Commerce and Industry",
        "source_url": "https://commerce.gov.in/trade-statistics",
        "released_on": "2025-08-11T00:00:00",
        "updated_on": "2025-08-11T00:00:00",
        "business_metadata": "Monthly trade data for policy analysis",
        "table_name": "trade_india_yr_index"  # Optional
    }
)
print(response.json())
```

### Option C: Using Swagger UI (Easiest!)
1. Go to `http://localhost:8000/docs`
2. Find `POST /approve/{job_id}`
3. Click "Try it out"
4. Fill in the form fields:
   - **job_id**: `abc123-def456-ghi789`
   - **table_name**: `trade_india_yr_index` (optional)
   - **source**: `Ministry of Commerce and Industry` ✅ Required
   - **source_url**: `https://commerce.gov.in/trade-statistics` ✅ Required
   - **released_on**: `2025-08-11T00:00:00` ✅ Required
   - **updated_on**: `2025-08-11T00:00:00` ✅ Required
   - **business_metadata**: `Monthly trade data for policy analysis` (optional)
5. Click "Execute"

**Response:**
```json
{
  "message": "Preprocessing approved. Database insertion started.",
  "job_id": "abc123-def456-ghi789",
  "table_name": "trade_india_yr_index",
  "status": "approved"
}
```

---

## 4. Check Final Status
```bash
curl "http://localhost:8000/status/abc123-def456-ghi789"
```

**Response:**
```json
{
  "job_id": "abc123-def456-ghi789",
  "status": "completed",
  "created_at": "2026-01-09T16:00:00",
  "updated_at": "2026-01-09T16:01:00",
  "result": {
    "table_name": "trade_india_yr_index",
    "rows_inserted": 1572,
    "columns": ["year", "month", "exports", "imports", ...],
    "processed_file_path": "processed/20260109_160100_trade_india_yr_index.csv",
    "warnings": []
  }
}
```

---

## What Happens Behind the Scenes

When you approve with metadata, the system:

1. **Inserts data** into `trade_india_yr_index` table
2. **Inserts metadata** into `tables_metadata`:
   - data_domain: "IIP" (from LLM)
   - comments: "Trade statistics for India..." (from LLM)
   - source: "Ministry of Commerce and Industry" (from you)
   - source_url: "https://commerce.gov.in..." (from you)
   - released_on, updated_on (from you)
   - period_cols: "year" (from LLM)
   - min_period_sql, max_period_sql (generated)

3. **Inserts metadata** into `operational_metadata`:
   - major_domain: "Trade and Commerce" (from LLM)
   - sub_domain: "International Trade" (from LLM)
   - brief_summary: "Trade statistics..." (from LLM)
   - first_available_value, last_available_value (extracted from data)
   - All user-provided fields

---

## Verify Metadata in Database

```sql
-- Check tables_metadata
SELECT * FROM tables_metadata WHERE table_name = 'trade_india_yr_index';

-- Check operational_metadata
SELECT * FROM operational_metadata WHERE table_name = 'trade_india_yr_index';
```

---

## Benefits of Individual Fields

✅ **Easier to use** - No need to construct JSON objects
✅ **Swagger UI friendly** - Nice form interface
✅ **Clear validation** - Each field validated separately
✅ **Better documentation** - Each parameter documented in API docs
