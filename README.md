# AI-Driven SQL Ingestion Agent

An intelligent data ingestion pipeline that uses OpenAI's LLM to analyze Excel/CSV files, preprocess data, and insert into PostgreSQL with human-in-the-loop approval.

## Features

- 🤖 **LLM-Powered Analysis**: Uses OpenAI to intelligently analyze file structure and generate table names
- 📊 **Smart Preprocessing**: Automatically merges headers and transforms date columns
- ✅ **Human-in-the-Loop**: Preview processed data before database insertion
- 🗄️ **Auto Schema Creation**: Dynamically creates PostgreSQL tables
- 📝 **Comprehensive Logging**: Detailed logs at each step in `.log` files
- 🔄 **Async Processing**: Non-blocking background tasks for file processing

## Architecture

```
Upload File → LLM Analysis → Preprocessing → Preview → User Approval → Save CSV → Create Table → Insert Data
```

## Installation

### Prerequisites

- Python 3.10+
- PostgreSQL database
- OpenAI API key

### Setup

1. **Clone or navigate to the project directory**

```bash
cd simple_sql_ingestion
```

2. **Create virtual environment**

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Configure environment variables**

Copy `.env.example` to `.env` and fill in your credentials:

```bash
copy .env.example .env  # Windows
# cp .env.example .env  # Linux/Mac
```

Edit `.env`:
```env
OPENAI_API_KEY=sk-your-actual-api-key
OPENAI_MODEL=gpt-4o-mini

POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=your_database
POSTGRES_USER=your_user
POSTGRES_PASSWORD=your_password

LOG_LEVEL=INFO
APPROVAL_TIMEOUT_MINUTES=30
```

## Usage

### Start the Server

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

### API Documentation

Interactive API docs: `http://localhost:8000/docs`

### Workflow

#### 1. Upload File

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@your_data.xlsx"
```

Response:
```json
{
  "job_id": "abc-123-def-456",
  "status": "preprocessing",
  "message": "File uploaded successfully. Preprocessing started."
}
```

#### 2. Check Status & Get Preview

```bash
curl "http://localhost:8000/status/abc-123-def-456"
```

When status is `awaiting_approval`, you'll get:
```json
{
  "job_id": "abc-123-def-456",
  "status": "awaiting_approval",
  "preview": {
    "proposed_table_name": "IIP_INDIA_MTH_SCTG",
    "columns": [
      {"name": "state", "type": "VARCHAR(100)"},
      {"name": "period", "type": "DATE"},
      {"name": "value", "type": "NUMERIC(10,2)"}
    ],
    "sample_rows": [
      {"state": "Maharashtra", "period": "2023-01-01", "value": 125.5},
      ...
    ],
    "total_rows": 1500,
    "preprocessing_summary": "Transformed 12 date columns to rows | Applied data cleaning"
  }
}
```

#### 3. Approve (or Reject)

**Approve with proposed table name:**
```bash
curl -X POST "http://localhost:8000/approve/abc-123-def-456"
```

**Approve with custom table name:**
```bash
curl -X POST "http://localhost:8000/approve/abc-123-def-456" \
  -H "Content-Type: application/json" \
  -d '{"table_name": "CUSTOM_TABLE_NAME"}'
```

**Reject:**
```bash
curl -X POST "http://localhost:8000/reject/abc-123-def-456"
```

#### 4. Check Completion

```bash
curl "http://localhost:8000/status/abc-123-def-456"
```

When status is `completed`:
```json
{
  "job_id": "abc-123-def-456",
  "status": "completed",
  "result": {
    "table_name": "IIP_INDIA_MTH_SCTG",
    "rows_inserted": 1500,
    "columns": ["state", "period", "value"],
    "processed_file_path": "processed/20260107_123456_IIP_INDIA_MTH_SCTG.csv",
    "warnings": []
  }
}
```

## Table Naming Convention

The LLM follows this convention: `<domain>_<Geo>_<Time>_<grain-Dimension>`

**Examples:**
- `IIP_India_Mth_SCtg` - IIP India Monthly SubCategory
- `IIP_State_Mth_Catg` - IIP Statewise Monthly Category
- `GDP_India_Qtr_Sector` - GDP India Quarterly Sector

**Components:**
- **domain**: Data domain (IIP, GDP, CPI, etc.)
- **Geo**: Geographic level (India, State, District)
- **Time**: Time granularity (Mth, Qtr, Yr)
- **grain-Dimension**: Data grain (Catg, SCtg, Sector)

## Project Structure

```
simple_sql_ingestion/
├── app/
│   ├── main.py              # FastAPI application
│   ├── models.py            # Pydantic models
│   ├── config.py            # Configuration
│   └── core/
│       ├── llm_architect.py # LLM intelligence
│       ├── preprocessor.py  # Data transformation
│       ├── database.py      # PostgreSQL operations
│       ├── logger.py        # Logging setup
│       └── job_manager.py   # Job state tracking
├── uploads/                 # Temporary uploaded files
├── processed/               # Processed CSV files (after approval)
├── logs/                    # Log files
├── requirements.txt
├── .env.example
└── README.md
```

## Logging

All operations are logged to `logs/ingestion.log` with:
- Timestamps
- Log levels (INFO, WARNING, ERROR)
- Function names and line numbers
- Detailed step-by-step processing information

## Error Handling

- Invalid file types are rejected
- LLM failures fall back to safe defaults
- Database errors are caught and logged
- Jobs expire after 30 minutes (configurable)

## Development

### Run with auto-reload

```bash
uvicorn app.main:app --reload --log-level debug
```

### View logs in real-time

```bash
tail -f logs/ingestion.log  # Linux/Mac
Get-Content logs/ingestion.log -Wait  # Windows PowerShell
```

## License

MIT
