from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime


class JobStatus(str, Enum):
    """Job processing status."""
    PREPROCESSING = "preprocessing"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    COMPLETED = "completed"
    FAILED = "failed"
    # Incremental Load statuses
    SIMILARITY_SEARCH = "similarity_search"
    INCREMENTAL_LOAD_AUTO = "incremental_load_auto"  # Auto-executing IL
    SCHEMA_MISMATCH = "schema_mismatch"  # Requires human approval
    INCREMENTAL_LOAD_COMPLETED = "incremental_load_completed"
    DUPLICATE_DATA_DETECTED = "duplicate_data_detected"  # Duplicate period data detected


class UploadResponse(BaseModel):
    """Response model for file upload endpoint."""
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Initial job status")
    message: str = Field(..., description="Status message")


class ColumnInfo(BaseModel):
    """Information about a single column."""
    name: str = Field(..., description="Column name")
    type: str = Field(..., description="PostgreSQL data type")
    sample_values: Optional[List[Any]] = Field(None, description="Sample values from the column")


class LLMMetadata(BaseModel):
    """LLM-generated metadata for preview."""
    suggested_domain: str = Field(..., description="LLM-classified data domain")
    description: str = Field(..., description="LLM-generated table description")
    period_column: Optional[str] = Field(None, description="Detected period/time column")


class SimilarTableMatch(BaseModel):
    """Information about a similar table found via similarity search."""
    table_name: str = Field(..., description="Name of the similar table")
    similarity_score: float = Field(..., description="Cosine similarity score (0-1)")
    columns: List[str] = Field(..., description="Columns in the existing table")
    row_count: int = Field(..., description="Current row count in the table")
    created_at: str = Field(..., description="When the signature was created")


class SchemaValidationResult(BaseModel):
    """Result of schema validation between new file and existing table."""
    is_compatible: bool = Field(..., description="Whether schemas are compatible")
    match_percentage: float = Field(..., description="Percentage of matching columns")
    matching_columns: List[str] = Field(..., description="Columns present in both")
    missing_columns: List[str] = Field(..., description="Columns in table but not in file")
    extra_columns: List[str] = Field(..., description="Columns in file but not in table")
    discrepancy_report: str = Field(..., description="Human-readable report")


class IncrementalLoadPreview(BaseModel):
    """Preview data for incremental load approval."""
    matched_table: SimilarTableMatch = Field(..., description="The matched table")
    validation_result: SchemaValidationResult = Field(..., description="Schema validation result")
    new_rows_count: int = Field(..., description="Number of rows to be added")
    current_rows_count: int = Field(..., description="Current rows in table")
    total_rows_after: int = Field(..., description="Total rows after incremental load")


class PreviewData(BaseModel):
    """Preview data for user approval."""
    proposed_table_name: str = Field(..., description="LLM-generated table name")
    columns: List[ColumnInfo] = Field(..., description="Column information with types")
    sample_rows: List[Dict[str, Any]] = Field(..., description="First 5 rows of processed data")
    total_rows: int = Field(..., description="Total number of rows in processed data")
    preprocessing_summary: str = Field(..., description="Summary of preprocessing steps applied")
    llm_metadata: Optional[LLMMetadata] = Field(None, description="LLM-generated metadata suggestions")


class MetadataInput(BaseModel):
    """User-provided metadata for approval."""
    table_name: Optional[str] = Field(None, description="Optional override for LLM-generated table name")
    source: str = Field(..., description="Data source name (e.g., 'NPCI', 'RBI')")
    source_url: str = Field(..., description="URL to data source")
    released_on: str = Field(..., description="Data release date (flexible format)")
    updated_on: str = Field(..., description="Last update date (flexible format)")
    business_metadata: Optional[str] = Field(None, description="Business context and notes")


class ApprovalRequest(BaseModel):
    """Request model for approval endpoint."""
    table_name: Optional[str] = Field(None, description="Optional override for table name")
    metadata: Optional[MetadataInput] = Field(None, description="Metadata for tables_metadata table")


class ProcessingResult(BaseModel):
    """Final processing results."""
    table_name: str = Field(..., description="Final table name used")
    rows_inserted: int = Field(..., description="Number of rows inserted into database")
    columns: List[str] = Field(..., description="List of column names")
    processed_file_path: str = Field(..., description="Path to saved processed CSV file")
    warnings: List[str] = Field(default_factory=list, description="Any warnings during processing")


class DuplicateDetectionResult(BaseModel):
    """Duplicate data detection results."""
    status: str = Field(..., description="NEW_DATA, DUPLICATE, PARTIAL_OVERLAP, or NO_PERIOD_COLUMN")
    message: str = Field(..., description="Human-readable message")
    existing_last_value: Optional[str] = Field(None, description="Last period value in existing table")
    new_first_value: Optional[str] = Field(None, description="First period value in new file")
    new_last_value: Optional[str] = Field(None, description="Last period value in new file")
    period_column: Optional[str] = Field(None, description="Name of the period column")


class StatusResponse(BaseModel):
    """Response model for status endpoint."""
    job_id: str = Field(..., description="Job identifier")
    status: JobStatus = Field(..., description="Current job status")
    created_at: str = Field(..., description="Job creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    preview: Optional[PreviewData] = Field(None, description="Preview data when awaiting approval")
    incremental_load_preview: Optional[IncrementalLoadPreview] = Field(None, description="IL preview when schema mismatch")
    duplicate_detection: Optional[DuplicateDetectionResult] = Field(None, description="Duplicate detection results")
    result: Optional[ProcessingResult] = Field(None, description="Processing result when completed")
    error: Optional[str] = Field(None, description="Error message if failed")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
