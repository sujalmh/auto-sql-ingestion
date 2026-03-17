from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from typing import Optional
import asyncio
import re
import urllib.parse
import pandas as pd
import shutil
from datetime import datetime

from app.models import (
    UploadResponse,
    StatusResponse,
    ApprovalRequest,
    ProcessingResult,
    PreviewData,
    ColumnInfo,
    ErrorResponse,
    JobStatus as JobStatusEnum,
    LLMMetadata,
    SimilarTableMatch,
    SchemaValidationResult,
    IncrementalLoadPreview
)
from app.core.logger import logger
from app.core.job_manager import job_manager, JobStatus
from app.core.llm_architect import llm_architect
from app.core.preprocessor import preprocessor
from app.core.database import db_manager
from app.core.metadata_generator import metadata_generator
from app.core.signature_builder import signature_builder
from app.core.milvus_manager import milvus_manager
from app.core.schema_validator import schema_validator
from app.core.incremental_loader import incremental_loader
from app.core.category_classifier import category_classifier
from app.core.cat2_preprocessor import cat2_preprocessor
from app.core.excel_analyzer import excel_analyzer, WorkbookStructure
from app.config import settings


# Initialize FastAPI app
app = FastAPI(
    title="AI-Driven SQL Ingestion Agent",
    description="Intelligent data ingestion pipeline with LLM-powered analysis and human-in-the-loop approval",
    version="1.0.0"
)

# Create necessary directories
UPLOAD_DIR = Path("uploads")
PROCESSED_DIR = Path("processed")
STATIC_DIR = Path(__file__).parent.parent / "static"
UPLOAD_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

# Serialization lock: ensures only one preprocess_file runs at a time.
# This is critical for peer-job matching in batch uploads — file B must
# wait for file A to finish so it can detect A as a peer match.
_preprocess_lock = asyncio.Lock()
_preprocess_semaphore = asyncio.Semaphore(5)

# Mount static files for web UI
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    logger.info("=" * 80)
    logger.info("AI-Driven SQL Ingestion Agent Starting")
    logger.info(f"OpenAI Model: {settings.openai_model}")
    logger.info(f"PostgreSQL Database: {settings.postgres_db}")
    logger.info(f"Approval Timeout: {settings.approval_timeout_minutes} minutes")
    logger.info("=" * 80)


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown."""
    logger.info("AI-Driven SQL Ingestion Agent Shutting Down")


@app.get("/")
async def root():
    """Serve the web UI homepage."""
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/admin/clear_jobs")
async def clear_jobs():
    """Clear all in-memory jobs. Use to start fresh."""
    n = job_manager.clear_all()
    return {"message": "All jobs cleared", "cleared": n}


@app.post("/upload", response_model=UploadResponse)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    file_description: Optional[str] = Form(None),
    table_name: Optional[str] = Form(None),
    skip_llm_table_name: bool = Form(False),
    sql_mode: Optional[str] = Form(None),
):
    """
    Upload a CSV or Excel file for processing.
    
    Args:
        file: The file to upload (CSV or Excel)
        file_description: Optional description/name to help generate better table name
                         (e.g., "Monthly GDP data", "Trade statistics")
    
    The file will be analyzed and preprocessed. Once ready, you'll receive
    a preview that requires approval before database insertion.
    """
    normalized_sql_mode = (sql_mode or "").strip().lower()
    logger.info(
        f"Received file upload: {file.filename}; sql_mode={normalized_sql_mode or 'none'}; "
        f"table_name={table_name or 'none'}; skip_llm_table_name={skip_llm_table_name}"
    )

    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    if skip_llm_table_name and table_name and normalized_sql_mode not in ("otl", "inc"):
        raise HTTPException(
            status_code=400,
            detail="sql_mode ('otl' or 'inc') is required when table_name override is used.",
        )

    if normalized_sql_mode in ("otl", "inc") and not table_name:
        raise HTTPException(
            status_code=400,
            detail="table_name is required when sql_mode is 'otl' or 'inc'.",
        )
    
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in settings.allowed_extensions:
        logger.warning(f"Invalid file extension: {file_ext}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {settings.allowed_extensions}"
        )
    
    try:
        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{file.filename}"
        file_path = UPLOAD_DIR / safe_filename
        
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"File saved to: {file_path}")
        
        # Create job
        job_id = job_manager.create_job(str(file_path))
        
        # Start preprocessing in background
        background_tasks.add_task(
            preprocess_file,
            job_id,
            str(file_path),
            file_description,
            table_name,
            skip_llm_table_name,
            normalized_sql_mode,
        )
        
        return UploadResponse(
            job_id=job_id,
            status=JobStatusEnum.PREPROCESSING,
            message=f"File uploaded successfully. Preprocessing started."
        )
        
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/status/{job_id}", response_model=StatusResponse)
async def get_status(job_id: str):
    """
    Get the current status of a processing job.
    
    When status is 'awaiting_approval', the response includes preview data
    with the first 5 rows for user review.
    """
    logger.info(f"Status check for job: {job_id}")
    
    job = job_manager.get_job(job_id)
    if not job:
        logger.warning(f"Job not found: {job_id}")
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Check if job has expired
    if job_manager.is_expired(job_id):
        logger.warning(f"Job {job_id} has expired")
        job_manager.update_status(job_id, JobStatus.FAILED)
        job_manager.set_error(job_id, "Job expired - approval timeout exceeded")
    
    # Build response based on status
    response = StatusResponse(
        job_id=job.job_id,
        status=JobStatusEnum(job.status.value),
        created_at=job.created_at.isoformat(),
        updated_at=job.updated_at.isoformat()
    )
    
    # Add preview data if awaiting approval
    if job.status == JobStatus.AWAITING_APPROVAL and job.processed_df is not None:
        # Get first 5 rows for preview
        preview_df = job.processed_df.head(5)
        
        # Build column info
        columns_info = []
        for col in job.processed_df.columns:
            col_type = job.column_types.get(col, "TEXT")
            # Safely access column data
            col_data = job.processed_df[col]
            if isinstance(col_data, pd.DataFrame):
                col_data = col_data.iloc[:, 0]
            sample_vals = col_data.dropna().head(3).tolist()
            columns_info.append(
                ColumnInfo(
                    name=col,
                    type=col_type,
                    sample_values=sample_vals
                )
            )
        
        # Build LLM metadata for preview
        llm_metadata_obj = None
        if job.llm_metadata:
            llm_metadata_obj = LLMMetadata(
                suggested_domain=job.llm_metadata.get('suggested_domain', 'Unknown'),
                description=job.llm_metadata.get('description', ''),
                period_column=job.llm_metadata.get('period_column')
            )
        
        response.preview = PreviewData(
            proposed_table_name=job.proposed_table_name,
            columns=columns_info,
            sample_rows=preview_df.to_dict(orient='records'),
            total_rows=len(job.processed_df),
            preprocessing_summary=job.preprocessing_summary or "No preprocessing applied",
            llm_metadata=llm_metadata_obj
        )
    
    # Add incremental load preview if schema mismatch
    elif job.status == JobStatus.SCHEMA_MISMATCH and job.schema_validation is not None:
        # Build similar table match info
        matched_table = job.matched_table_name
        similarity_score = job.similar_tables[0]['similarity_score'] if job.similar_tables else 0.0
        
        # Fetch metadata for matched table
        matched_metadata = schema_validator.fetch_table_metadata(matched_table)
        
        if matched_metadata:
            similar_table_match = SimilarTableMatch(
                table_name=matched_table,
                similarity_score=similarity_score,
                columns=matched_metadata['columns'],
                row_count=matched_metadata['rows_count'],
                created_at=job.similar_tables[0]['created_at'] if job.similar_tables else ""
            )
            
            # Build schema validation result
            schema_val_result = SchemaValidationResult(
                is_compatible=job.schema_validation['is_compatible'],
                match_percentage=job.schema_validation['match_percentage'],
                matching_columns=job.schema_validation['matching_columns'],
                missing_columns=job.schema_validation['missing_columns'],
                extra_columns=job.schema_validation['extra_columns'],
                discrepancy_report=schema_validator.generate_discrepancy_report(
                    job.schema_validation,
                    matched_table,
                    Path(job.file_path).name
                )
            )
            
            # Get current row count
            current_rows = incremental_loader.get_current_row_count(matched_table) or matched_metadata['rows_count']
            new_rows = len(job.processed_df) if job.processed_df is not None else 0
            
            response.incremental_load_preview = IncrementalLoadPreview(
                matched_table=similar_table_match,
                validation_result=schema_val_result,
                new_rows_count=new_rows,
                current_rows_count=current_rows,
                total_rows_after=current_rows + new_rows
            )
    
    # Add duplicate detection results if available
    if job.duplicate_detection and job.status in [JobStatus.DUPLICATE_DATA_DETECTED, JobStatus.SCHEMA_MISMATCH]:
        from app.models import DuplicateDetectionResult
        response.duplicate_detection = DuplicateDetectionResult(**job.duplicate_detection)
    
    # Add result if completed
    elif job.status == JobStatus.COMPLETED:
        response.result = ProcessingResult(
            table_name=job.final_table_name,
            rows_inserted=job.rows_inserted,
            columns=job.processed_df.columns.tolist() if job.processed_df is not None else [],
            processed_file_path=job.processed_file_path or "",
            warnings=job.warnings
        )
    
    # Add error if failed
    elif job.status == JobStatus.FAILED:
        response.error = job.error
    
    return response


@app.get("/job/{job_id}/export")
async def export_job_data(job_id: str, format: str = "csv"):
    """
    Export the full processed table for a job (CSV or JSON).
    Available when job has processed data (e.g. awaiting_approval, schema_mismatch).
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.processed_df is None:
        raise HTTPException(status_code=404, detail="No processed data for this job")
    table_name = (job.proposed_table_name or job_id).replace(" ", "_")
    if format == "csv":
        content = job.processed_df.to_csv(index=False).encode("utf-8")
        return Response(
            content=content,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={table_name}.csv"},
        )
    if format == "json":
        # to_json produces valid JSON (NaN -> null, dates as ISO)
        content = job.processed_df.to_json(orient="records", date_format="iso")
        return Response(content=content, media_type="application/json")
    raise HTTPException(status_code=400, detail="format must be 'csv' or 'json'")


@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    """
    Completely delete all information associated with a job from disk and database.
    """
    logger.info(f"Deletion request for job: {job_id}")
    
    job = job_manager.get_job(job_id)
    if not job:
        logger.warning(f"Job not found for deletion: {job_id}")
        raise HTTPException(status_code=404, detail="Job not found")
    
    deleted_items = []
    errors = []

    # 1. Delete original uploaded file
    try:
        if job.file_path:
            file_path = Path(job.file_path)
            if file_path.exists():
                file_path.unlink()
                deleted_items.append(f"Uploaded file: {file_path.name}")
    except Exception as e:
        errors.append(f"Error deleting uploaded file: {str(e)}")

    # 2. Delete processed CSV file (handle partial completion via globbing)
    try:
        table_name_to_clean = job.final_table_name or job.table_name or job.proposed_table_name
        
        # Exact match if it exists
        if job.processed_file_path:
            process_path = Path(job.processed_file_path)
            if process_path.exists():
                process_path.unlink()
                deleted_items.append(f"Processed file: {process_path.name}")
                
        # Search for dangling/partial files matching table name
        if table_name_to_clean:
            for p in PROCESSED_DIR.glob(f"*_{table_name_to_clean}*.csv"):
                p.unlink()
                deleted_items.append(f"Processed file (partial): {p.name}")
    except Exception as e:
        errors.append(f"Error deleting processed file: {str(e)}")

    # 3. Delete database table and metadata
    try:
        table_name = job.final_table_name or job.table_name or job.proposed_table_name
        if table_name:
            if db_manager.delete_table_and_metadata(table_name):
                deleted_items.append(f"Database table & metadata: {table_name}")
            else:
                errors.append(f"Could not completely delete DB records for {table_name}")
    except Exception as e:
        errors.append(f"Error deleting database records: {str(e)}")

    # 4. Remove job from memory
    if job_manager.delete_job(job_id):
        deleted_items.append(f"In-memory job record: {job_id}")

    if errors:
        return {"message": "Job deleted with some errors", "deleted": deleted_items, "errors": errors}
    
    return {"message": "Job completely deleted", "deleted": deleted_items}


@app.post("/approve/{job_id}")
async def approve_job(
    job_id: str,
    background_tasks: BackgroundTasks,
    table_name: Optional[str] = Form(None),
    source: str = Form(...),
    source_url: str = Form(...),
    released_on: str = Form(...),
    updated_on: str = Form(...),
    business_metadata: Optional[str] = Form(None)
):
    """
    Approve the preprocessing results and trigger database insertion.
    
    Args:
        job_id: Job identifier (in URL path)
        table_name: Optional override for table name
        source: Data source name (required)
        source_url: URL to data source (required)
        released_on: Release date in ISO format (required)
        updated_on: Update date in ISO format (required)
        business_metadata: Business context (optional)
    """
    logger.info(f"Approval request for job: {job_id}")
    
    job = job_manager.get_job(job_id)
    if not job:
        logger.warning(f"Job not found: {job_id}")
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status not in [JobStatus.AWAITING_APPROVAL, JobStatus.SCHEMA_MISMATCH, JobStatus.DUPLICATE_DATA_DETECTED]:
        logger.warning(f"Job {job_id} not in approvable status: {job.status}")
        raise HTTPException(
            status_code=400,
            detail=f"Job is not awaiting approval. Current status: {job.status.value}"
        )
    
    # Parse dates flexibly
    from dateutil import parser as date_parser
    try:
        released_on_parsed = date_parser.parse(released_on).isoformat()
        updated_on_parsed = date_parser.parse(updated_on).isoformat()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {str(e)}")
    
    # Create MetadataInput object from individual fields
    try:
        from app.models import MetadataInput
        metadata = MetadataInput(
            source=source,
            source_url=source_url,
            released_on=released_on_parsed,
            updated_on=updated_on_parsed,
            business_metadata=business_metadata
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid metadata: {str(e)}")
    
    # Use custom table name if provided, otherwise use proposed name
    final_table_name = table_name or job.proposed_table_name
    # Ensure table name is lowercase for PostgreSQL consistency
    final_table_name = final_table_name.lower()
    
    # Save the target table name immediately for partial completion cleanup
    job.table_name = final_table_name
    
    logger.info(f"Job {job_id} approved with table name: {final_table_name}")
    
    # Update status to approved
    job_manager.update_status(job_id, JobStatus.APPROVED)
    
    # Route to OTL or IL based on job status
    if job.is_incremental_load and job.matched_table_name:
        # Peer-job guard: if this IL was matched against a peer job that hasn't
        # been committed yet, block and tell the user to approve the OTL first.
        if job.peer_anchor_job_id:
            anchor_job = job_manager.get_job(job.peer_anchor_job_id)
            if anchor_job and anchor_job.status not in (
                JobStatus.COMPLETED,
                JobStatus.INCREMENTAL_LOAD_COMPLETED,
            ):
                anchor_desc = (
                    anchor_job.proposed_table_name
                    or Path(anchor_job.file_path).stem
                )
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"This incremental load targets table '{job.matched_table_name}' "
                        f"from a batch peer job (job {job.peer_anchor_job_id[:8]}... "
                        f"- '{anchor_desc}') that hasn't been approved yet. "
                        f"Please approve the OTL job first, then approve this one."
                    ),
                )

        # Incremental Load workflow
        logger.info(f"Job {job_id} approved for incremental load to table: {job.matched_table_name}")
        background_tasks.add_task(perform_incremental_load, job_id, job.matched_table_name, metadata)
        
        return {
            "message": "Incremental load approved. Data appending started.",
            "job_id": job_id,
            "table_name": job.matched_table_name,
            "load_type": "incremental",
            "status": "approved"
        }
    else:
        # One-Time Load workflow
        logger.info(f"Job {job_id} approved for one-time load with table: {final_table_name}")
        background_tasks.add_task(insert_to_database, job_id, final_table_name, metadata)
        
        return {
            "message": "Preprocessing approved. Database insertion started.",
            "job_id": job_id,
            "table_name": final_table_name,
            "load_type": "one_time",
            "status": "approved"
        }


@app.post("/reject/{job_id}")
async def reject_job(job_id: str):
    """
    Reject the preprocessing results and cleanup temporary files.
    """
    logger.info(f"Rejection request for job: {job_id}")
    
    job = job_manager.get_job(job_id)
    if not job:
        logger.warning(f"Job not found: {job_id}")
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Update status to rejected
    job_manager.update_status(job_id, JobStatus.REJECTED)
    
    # Cleanup uploaded file
    try:
        file_path = Path(job.file_path)
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Deleted uploaded file: {file_path}")
    except Exception as e:
        logger.error(f"Error deleting file: {str(e)}")
    
    return {
        "message": "Job rejected and cleaned up.",
        "job_id": job_id,
        "status": "rejected"
    }


@app.get("/pending-jobs")
async def list_pending_jobs():
    """
    List all jobs that are awaiting approval (OTL, IL schema-mismatch,
    or duplicate-data-detected).  Returns a lightweight summary per job
    suitable for the batch-approve UI.
    """
    pending = job_manager.get_pending_jobs()
    results = []
    for job in pending:
        load_type = "IL" if (job.is_incremental_load and job.matched_table_name) else "OTL"
        table_name = job.matched_table_name if load_type == "IL" else job.proposed_table_name
        total_rows = len(job.processed_df) if job.processed_df is not None else 0
        columns_count = len(job.processed_df.columns) if job.processed_df is not None else 0
        filename = Path(job.file_path).name if job.file_path else ""
        results.append({
            "job_id": job.job_id,
            "filename": filename,
            "proposed_table_name": table_name,
            "load_type": load_type,
            "status": job.status.value,
            "total_rows": total_rows,
            "columns_count": columns_count,
            "created_at": job.created_at.isoformat(),
        })
    return {"pending_jobs": results}


@app.post("/batch-approve")
async def batch_approve(
    background_tasks: BackgroundTasks,
    source: str = Form(...),
    source_url: str = Form(...),
    released_on: str = Form(...),
    updated_on: str = Form(...),
    business_metadata: Optional[str] = Form(None),
):
    """
    Approve **all** pending jobs at once with shared metadata.

    The same source / source_url / dates are applied to every job that
    is currently in an approvable status.
    """
    from dateutil import parser as date_parser
    from app.models import MetadataInput

    try:
        released_on_parsed = date_parser.parse(released_on).isoformat()
        updated_on_parsed = date_parser.parse(updated_on).isoformat()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {str(e)}")

    metadata = MetadataInput(
        source=source,
        source_url=source_url,
        released_on=released_on_parsed,
        updated_on=updated_on_parsed,
        business_metadata=business_metadata,
    )

    pending = job_manager.get_pending_jobs()
    if not pending:
        raise HTTPException(status_code=404, detail="No pending jobs to approve")

    # Sort so OTL jobs are approved before their IL peers.
    # OTL jobs have is_incremental_load=False; IL jobs True.
    pending.sort(key=lambda j: (j.is_incremental_load, j.created_at))

    approved: list[dict] = []
    errors: list[dict] = []

    for job in pending:
        job_id = job.job_id
        is_il = job.is_incremental_load and job.matched_table_name

        # Peer-job guard for IL: skip if anchor not yet committed (it
        # will be committed by the time the background task runs, since
        # OTL jobs are sorted first, but validate anyway).
        if is_il and job.peer_anchor_job_id:
            anchor = job_manager.get_job(job.peer_anchor_job_id)
            if anchor and anchor.status not in (
                JobStatus.COMPLETED,
                JobStatus.INCREMENTAL_LOAD_COMPLETED,
                JobStatus.APPROVED,
            ):
                errors.append({
                    "job_id": job_id,
                    "error": (
                        f"Peer anchor job {job.peer_anchor_job_id[:8]}... "
                        f"not yet approved/completed. Skipped."
                    ),
                })
                continue

        final_table_name = job.matched_table_name if is_il else (job.proposed_table_name or "").lower()
        job.table_name = final_table_name
        job_manager.update_status(job_id, JobStatus.APPROVED)

        if is_il:
            background_tasks.add_task(
                perform_incremental_load, job_id, job.matched_table_name, metadata
            )
        else:
            background_tasks.add_task(
                insert_to_database, job_id, final_table_name, metadata
            )

        approved.append({
            "job_id": job_id,
            "table_name": final_table_name,
            "load_type": "incremental" if is_il else "one_time",
        })

    return {
        "message": f"Batch approved {len(approved)} job(s).",
        "approved": approved,
        "errors": errors,
    }


# Background task functions

async def preprocess_file(
    job_id: str,
    file_path: str,
    file_description: Optional[str] = None,
    table_name_override: Optional[str] = None,
    skip_llm_table_name: bool = False,
    sql_mode: Optional[str] = None,
):
    """
    Background task to preprocess the uploaded file.
    
    Phase A (concurrent, semaphore-limited to 5):
      Steps 1-8: file loading, LLM analysis, preprocessing, type inference,
      metadata generation, Milvus signature + search.
    
    Phase B (sequential, under lock):
      Steps 9-10: IL resolution (Milvus match validation, column fallback,
      peer-job matching, OTL determination), store results.
    """
    try:
        # Phase A: concurrent preprocessing (limited to 5 parallel)
        async with _preprocess_semaphore:
            phase_a_ctx = await _preprocess_phase_a(
                job_id, file_path, file_description,
                table_name_override, skip_llm_table_name, sql_mode,
            )
        if phase_a_ctx is None:
            return  # error already set inside phase_a

        # Phase B: sequential IL resolution
        async with _preprocess_lock:
            await _preprocess_phase_b(job_id, file_path, file_description, phase_a_ctx)
    except Exception as e:
        logger.error(f"[Job {job_id}] Preprocessing failed: {str(e)}", exc_info=True)
        job_manager.set_error(job_id, f"Preprocessing failed: {str(e)}")


def _generate_non_colliding_table_name_via_llm(
    df: pd.DataFrame,
    analysis: dict,
    file_description: Optional[str],
    filename_stem: str,
    existing_name: str,
) -> str:
    """
    Generate a different, meaningful table name when LLM collides with
    an existing DB table in a non-incremental flow.
    """
    for _ in range(3):
        collision_hint = (
            (file_description or "")
            + f" Existing table name '{existing_name}' already exists. "
            + "Generate a DIFFERENT name for a new one-time-load dataset. "
            + "Do not return the existing table name."
        ).strip()

        candidate = llm_architect.generate_table_name(
            df,
            analysis,
            collision_hint,
            filename=filename_stem,
        )
        candidate = llm_architect.refine_table_name(candidate)

        if candidate and candidate != existing_name and not db_manager.table_exists(candidate):
            return candidate

    # Final fallback: still non-numeric-sequence naming, uniquely timestamped.
    fallback = f"{existing_name}_new_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return llm_architect.refine_table_name(fallback)


def _load_sub_table_df(file_path: str, sub, job_id: str) -> pd.DataFrame:
    """Load a single SubTable from an Excel workbook as a DataFrame."""

    # ── Horizontal sub-tables: skip all header rows, load raw data + usecols,
    #    then apply pre-computed column names (pandas forbids usecols + multi-header).
    if getattr(sub, 'horizontal_split', False):
        all_skiprows = list(range(0, sub.data_start_row - 1))
        if sub.col_number_row is not None:
            all_skiprows.append(sub.col_number_row - 1)
        usecols = list(range(sub.min_data_col - 1, sub.max_data_col))
        nrows = sub.data_end_row - sub.data_start_row + 1

        df = pd.read_excel(
            file_path,
            sheet_name=sub.sheet_name,
            skiprows=sorted(set(all_skiprows)),
            header=None,
            nrows=nrows,
            usecols=usecols,
        )
        # Apply pre-computed column names
        if sub.merged_header_values and sub.merged_header_values[0]:
            names = sub.merged_header_values[0]
            if len(names) == len(df.columns):
                df.columns = [n or f"column_{i}" for i, n in enumerate(names)]

        logger.info(
            f"[Job {job_id}] Loaded horizontal sub-table [{sub.sheet_name}] "
            f"rows {sub.data_start_row}-{sub.data_end_row}, "
            f"cols {sub.min_data_col}-{sub.max_data_col}: "
            f"{df.shape[0]}x{df.shape[1]}, label={sub.label}"
        )
        return df

    # ── Normal (full-width) sub-table loading ──
    # Skip all rows before the first header row (0-indexed)
    skiprows = list(range(0, sub.header_rows[0] - 1))
    # Also skip the column-numbering row if present
    if sub.col_number_row is not None:
        skiprows.append(sub.col_number_row - 1)

    header_param = 0 if sub.num_header_rows == 1 else list(range(sub.num_header_rows))
    nrows = sub.data_end_row - sub.data_start_row + 1

    df = pd.read_excel(
        file_path,
        sheet_name=sub.sheet_name,
        skiprows=skiprows,
        header=header_param,
        nrows=nrows,
    )

    # Drop fully empty leading columns
    while len(df.columns) > 0:
        first_col = df.iloc[:, 0]
        if first_col.isna().all() or (first_col.astype(str).str.strip() == '').all():
            df = df.iloc[:, 1:]
        else:
            break

    # Replace broken MultiIndex with merged header names
    if sub.num_header_rows > 1 and sub.merged_header_values and isinstance(df.columns, pd.MultiIndex):
        merged_names = []
        for col_idx in range(len(df.columns)):
            parts = []
            seen = set()
            for level in sub.merged_header_values:
                if col_idx < len(level) and level[col_idx]:
                    val = level[col_idx]
                    if val not in seen:
                        parts.append(val)
                        seen.add(val)
            merged_names.append("_".join(parts) if parts else f"column_{col_idx}")
        df.columns = merged_names

    logger.info(
        f"[Job {job_id}] Loaded sub-table [{sub.sheet_name}] "
        f"rows {sub.data_start_row}-{sub.data_end_row}: "
        f"{df.shape[0]}x{df.shape[1]}, base_year={sub.base_year}, label={sub.label}"
    )
    return df


async def _preprocess_phase_a(
    job_id: str,
    file_path: str,
    file_description: Optional[str] = None,
    table_name_override: Optional[str] = None,
    skip_llm_table_name: bool = False,
    sql_mode: Optional[str] = None,
) -> Optional[dict]:
    logger.info(f"[Job {job_id}] Starting preprocessing pipeline")
    normalized_sql_mode = (sql_mode or "").strip().lower()
    if normalized_sql_mode in ("otl", "inc") and table_name_override:
        skip_llm_table_name = True
    logger.info(
        f"[Job {job_id}] Routing controls: sql_mode={normalized_sql_mode or 'none'}, "
        f"table_name_override={table_name_override or 'none'}, "
        f"skip_llm_table_name={skip_llm_table_name}"
    )
    
    try:
        table_name_generated_by_llm = False

        # Step 1: Smart header detection
        logger.info(f"[Job {job_id}] Loading file for header detection: {file_path}")
        
        excel_structure = None  # Will be set for .xlsx files
        
        if file_path.endswith('.csv'):
            # Robust CSV delimiter detection.
            # 1. Check for an explicit "sep=X" metadata directive on line 1
            #    (used by RBI and some other data portals, e.g. "sep=|").
            # 2. Fall back to a column-count consistency scorer across lines.
            detected_delimiter = ','
            skip_rows = 0
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    first_line = f.readline().rstrip('\n').rstrip('\r')
                    raw_lines = [first_line] + [f.readline() for _ in range(19)]
                raw_lines = [l for l in raw_lines if l.strip()]

                # Check for explicit "sep=X" directive (case-insensitive)
                if first_line.lower().startswith('sep=') and len(first_line.strip()) <= 6:
                    detected_delimiter = first_line.strip()[4:]  # char after "sep="
                    skip_rows = 1
                    logger.info(
                        f"[Job {job_id}] Found 'sep=' directive, delimiter={repr(detected_delimiter)}, skipping row 1"
                    )
                else:
                    # Consistency-based scoring: pick the delimiter whose column
                    # count is most stable (lowest variance) across lines.
                    def _delimiter_score(delim):
                        counts = [len(line.split(delim)) for line in raw_lines]
                        if not counts:
                            return 0, 0
                        avg = sum(counts) / len(counts)
                        if avg <= 1:
                            return 0, 0
                        variance = sum((c - avg) ** 2 for c in counts) / len(counts)
                        return avg / (1 + variance), avg

                    candidates = [',', '|', '\t', ';', ':']
                    best_delim, best_score = ',', -1
                    for delim in candidates:
                        score, _ = _delimiter_score(delim)
                        if score > best_score:
                            best_score = score
                            best_delim = delim
                    detected_delimiter = best_delim
                    logger.info(f"[Job {job_id}] Detected CSV delimiter: {repr(detected_delimiter)}")
            except Exception as e:
                logger.warning(f"[Job {job_id}] Could not detect delimiter, defaulting to comma: {e}")

            # Load first 5 rows without assuming header structure
            df_preview = pd.read_csv(
                file_path, header=None, nrows=5, sep=detected_delimiter, skiprows=skip_rows
            )
            
            # Ask LLM to detect header row count
            header_count = llm_architect.detect_header_rows(df_preview)
            logger.info(f"[Job {job_id}] Detected {header_count} header row(s)")
            
            # Re-load file with correct header parameter
            if header_count == 1:
                df = pd.read_csv(file_path, sep=detected_delimiter, skiprows=skip_rows)
            elif header_count == 2:
                df = pd.read_csv(file_path, header=[0, 1], sep=detected_delimiter, skiprows=skip_rows)
            else:  # header_count == 3
                df = pd.read_csv(file_path, header=[0, 1, 2], sep=detected_delimiter, skiprows=skip_rows)
        else:
            # ── Excel file: use openpyxl-based structural analysis ──
            # First check for multi-sheet / vertically-stacked sub-table patterns
            wb_structure = excel_analyzer.analyze_workbook(file_path)
            logger.info(
                f"[Job {job_id}] Workbook: {len(wb_structure.sub_tables)} sub-tables "
                f"across {len(wb_structure.sheet_names)} sheets, "
                f"mergeable={wb_structure.mergeable}"
            )

            if wb_structure.mergeable and len(wb_structure.sub_tables) > 1:
                # ── Multiple sub-tables with same columns → load each, add labels, concat ──
                _BY_RE = re.compile(r"(\d{4}(?:[–\-]\d{2,4})?)")
                dfs = []
                for sub in wb_structure.sub_tables:
                    sub_df = _load_sub_table_df(file_path, sub, job_id)
                    # Derive base_year from sheet name if not found in banners
                    if not sub.base_year:
                        by_m = _BY_RE.search(sub.sheet_name)
                        if by_m:
                            sub.base_year = by_m.group(1)
                    # Add distinguishing columns
                    if sub.base_year:
                        sub_df.insert(0, "base_year", sub.base_year)
                    if sub.label:
                        col_name = getattr(sub, 'label_column', None) or "price_type"
                        pos = 1 if "base_year" in sub_df.columns else 0
                        sub_df.insert(pos, col_name, sub.label)
                    # Fallback: use sheet name as distinguishing column
                    if not sub.base_year and not sub.label:
                        sub_df.insert(0, "sheet_label", sub.sheet_name)
                    dfs.append(sub_df)

                df = pd.concat(dfs, ignore_index=True)
                logger.info(
                    f"[Job {job_id}] Merged {len(dfs)} sub-tables: "
                    f"{df.shape[0]} rows x {df.shape[1]} columns"
                )
                # Use workbook-level hints for downstream metadata injection
                excel_structure = wb_structure
            else:
                # ── Single table (or non-mergeable): use original single-table flow ──
                # When sub-tables exist but aren't mergeable, load only the first
                # sub-table to avoid confusing the row classifier with mixed data.
                if wb_structure.sub_tables:
                    first_sub = wb_structure.sub_tables[0]
                    df = _load_sub_table_df(file_path, first_sub, job_id)
                    logger.info(
                        f"[Job {job_id}] Non-mergeable: loaded first sub-table "
                        f"[{first_sub.sheet_name}] rows {first_sub.data_start_row}-"
                        f"{first_sub.data_end_row}: {df.shape[0]}x{df.shape[1]}"
                    )
                    excel_structure = wb_structure
                else:
                    excel_structure = excel_analyzer.analyze(file_path)
                    es = excel_structure
                    logger.info(
                        f"[Job {job_id}] Excel structure: skip={es.skip_rows}, "
                        f"headers={es.header_rows}, col_num_row={es.col_number_row}, "
                        f"data={es.data_start_row}-{es.data_end_row}, "
                        f"footer={es.footer_start_row}"
                    )
                    
                    # Compute pandas parameters from ExcelStructure
                    pandas_skiprows = list(range(0, es.skip_rows))  # 0-indexed
                    
                    if es.col_number_row is not None:
                        pandas_skiprows.append(es.col_number_row - 1)
                    
                    nrows_param = None
                    if es.footer_start_row is not None:
                        total_rows = es.data_end_row - es.header_rows[0] + 1
                        if es.col_number_row is not None and es.header_rows[0] <= es.col_number_row <= es.data_end_row:
                            total_rows -= 1
                        total_rows -= es.num_header_rows
                        nrows_param = max(total_rows, 0)
                    
                    if es.num_header_rows == 1:
                        header_param = 0
                    else:
                        header_param = list(range(es.num_header_rows))
                    
                    logger.info(
                        f"[Job {job_id}] pd.read_excel params: skiprows={pandas_skiprows}, "
                        f"header={header_param}, nrows={nrows_param}"
                    )
                    
                    if nrows_param is not None:
                        df = pd.read_excel(
                            file_path,
                            skiprows=pandas_skiprows,
                            header=header_param,
                            nrows=nrows_param,
                        )
                    else:
                        df = pd.read_excel(
                            file_path,
                            skiprows=pandas_skiprows,
                            header=header_param,
                        )
                    
                    # Drop fully empty leading columns
                    while len(df.columns) > 0:
                        first_col = df.iloc[:, 0]
                        if first_col.isna().all() or (first_col.astype(str).str.strip() == '').all():
                            df = df.iloc[:, 1:]
                        else:
                            break
                    
                    # Replace broken MultiIndex with merged header names
                    if es.num_header_rows > 1 and es.merged_header_values and isinstance(df.columns, pd.MultiIndex):
                        merged_names = []
                        num_cols = len(df.columns)
                        header_grid = es.merged_header_values
                        for col_idx in range(num_cols):
                            parts = []
                            seen = set()
                            for level in header_grid:
                                if col_idx < len(level):
                                    val = level[col_idx]
                                    if val and val not in seen:
                                        parts.append(val)
                                        seen.add(val)
                            merged_names.append('_'.join(parts) if parts else f'column_{col_idx}')
                        df.columns = merged_names
                        logger.info(f"[Job {job_id}] Applied merged column names from Excel structure: {merged_names}")
        
        logger.info(f"[Job {job_id}] Loaded {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Step 2: Analyze file structure with LLM
        logger.info(f"[Job {job_id}] Analyzing file structure with LLM")
        analysis = llm_architect.analyze_file_structure(df)
        
        # Inject Excel-level metadata into analysis if available
        if excel_structure:
            if excel_structure.title_hint:
                analysis["excel_title"] = excel_structure.title_hint
            if excel_structure.unit_hint:
                analysis["excel_unit"] = excel_structure.unit_hint
        
        logger.info(f"[Job {job_id}] Analysis complete: {analysis}")
        
        # Step 2b: Classify as Category 1 or 2 (row-structure complexity)
        # Force Category 1 for merged sub-table DataFrames — their row structure
        # is already resolved by workbook-level analysis; Cat2 row classification
        # misidentifies repeated state/entity names as group headers.
        from_subtable = (
            isinstance(excel_structure, WorkbookStructure)
            and len(getattr(excel_structure, "sub_tables", [])) > 0
        )
        if from_subtable:
            classification = {"category": 1, "confidence": 1.0, "reasoning": "Forced Cat1 — sub-table boundaries already determined by workbook analysis"}
            data_category = 1
            logger.info(f"[Job {job_id}] Forced Category 1 for workbook with known sub-table structure")
        else:
            classification = category_classifier.classify(df, analysis)
            data_category = classification.get("category", 1)
        
        # Step 2c: Normalize filename for table naming
        # URL-decode, strip download suffixes (-M_p, -Y_p, _p), detect frequency
        raw_stem = Path(file_path).stem
        norm_stem = urllib.parse.unquote(raw_stem)
        # Detect monthly/yearly frequency from suffix before stripping
        _freq_suffix = ""
        if re.search(r'[-_]M_p$', norm_stem, re.IGNORECASE):
            _freq_suffix = "_monthly"
        elif re.search(r'[-_]Y_p$', norm_stem, re.IGNORECASE):
            _freq_suffix = "_yearly"
        # Strip download suffixes
        norm_stem = re.sub(r'[-_][MY]_p$', '', norm_stem, flags=re.IGNORECASE)
        norm_stem = re.sub(r'_p$', '', norm_stem, flags=re.IGNORECASE)
        # Strip leading "Table N " prefix
        norm_stem = re.sub(r'^Table\s*\d+\s*[-_]?\s*', '', norm_stem, flags=re.IGNORECASE)
        # Replace URL-encoding residuals and special chars
        norm_stem = re.sub(r'[%+]', ' ', norm_stem).strip()
        norm_stem = norm_stem + _freq_suffix
        logger.info(f"[Job {job_id}] Normalized filename: '{raw_stem}' → '{norm_stem}'")
        
        # Step 3: Determine table name
        if skip_llm_table_name and table_name_override:
            logger.info(f"[Job {job_id}] Skipping LLM table-name generation due to manual override")
            table_name = table_name_override.strip().lower().replace(" ", "_")
            logger.info(f"[Job {job_id}] Using manual table name: {table_name}")
        else:
            logger.info(f"[Job {job_id}] Generating table name with LLM")
            table_name = llm_architect.generate_table_name(df, analysis, file_description, filename=norm_stem)
            table_name_generated_by_llm = True

            # Step 3b: Refine table name if too long
            table_name = llm_architect.refine_table_name(table_name)
            logger.info(f"[Job {job_id}] Final table name: {table_name}")
        
        # Step 4: Preprocess data
        logger.info(f"[Job {job_id}] Preprocessing data")
        if data_category == 2:
            processed_df, summary = cat2_preprocessor.preprocess(df, analysis, classification)
        else:
            processed_df, summary = preprocessor.preprocess(df, analysis)
        logger.info(f"[Job {job_id}] Preprocessing complete: {summary}")
        
        # Step 4b: Clean column names with LLM (rename + drop redundant columns)
        logger.info(f"[Job {job_id}] Cleaning column names")
        original_columns = processed_df.columns.tolist()
        logger.info(f"[Job {job_id}] Original columns before cleaning: {original_columns}")

        sample_rows = processed_df.head(3).to_dict("records") if len(processed_df) > 0 else None
        clean_result = llm_architect.clean_column_names(
            original_columns, sample_rows=sample_rows
        )
        column_mapping = clean_result.get("mapping", {})
        drop_columns = clean_result.get("drop_columns", [])

        # Protect critical columns from being dropped or renamed by the LLM
        protected_columns = {'month_numeric', 'month_name', 'month', 'year', 'period',
                             'fiscal_year', 'quarter',
                             'from_month', 'from_year', 'to_month', 'to_year',
                             'state', 'value', 'total', 'refresh_date', 'data_period'}

        # Drop redundant/metadata columns first (with safeguard for critical columns)
        if drop_columns:
            protected_dropped = [c for c in drop_columns if c.lower() in protected_columns]
            if protected_dropped:
                logger.warning(f"[Job {job_id}] LLM tried to drop protected columns: {protected_dropped} — keeping them")
            drop_columns = [c for c in drop_columns if c.lower() not in protected_columns]
            drop_present = [c for c in drop_columns if c in processed_df.columns]
            if drop_present:
                processed_df = processed_df.drop(columns=drop_present)
                logger.info(f"[Job {job_id}] Dropped redundant/metadata columns: {drop_present}")

        # Apply cleaned column names (mapping only includes kept columns)
        if column_mapping:
            # Only rename columns that still exist; protect structural time columns
            # from being renamed to avoid losing from/to semantics.
            rename_map = {k: v for k, v in column_mapping.items()
                          if k in processed_df.columns and k.lower() not in protected_columns}
            processed_df.rename(columns=rename_map, inplace=True)
            logger.info(f"[Job {job_id}] Columns after LLM cleaning: {processed_df.columns.tolist()}")

            # Ensure unique column names (handle exact + case-insensitive collisions)
            cols = processed_df.columns.tolist()
            seen_normalized = {}
            new_cols = []
            duplicates_found = []

            def _norm_col_name(name: str) -> str:
                # Match DB identifier normalization rules relevant for duplicates.
                return str(name).replace('"', '').strip().lower()

            for idx, col in enumerate(cols):
                col_str = str(col)
                norm = _norm_col_name(col_str)
                if norm in seen_normalized:
                    seen_normalized[norm] += 1
                    suffix = str(seen_normalized[norm])
                    base = col_str.strip() or f"column_{idx}"
                    candidate = f"{base}_{suffix}"
                    candidate_norm = _norm_col_name(candidate)
                    while candidate_norm in seen_normalized:
                        seen_normalized[norm] += 1
                        suffix = str(seen_normalized[norm])
                        candidate = f"{base}_{suffix}"
                        candidate_norm = _norm_col_name(candidate)

                    new_cols.append(candidate)
                    seen_normalized[candidate_norm] = 0
                    duplicates_found.append((col_str, candidate, original_columns[idx] if idx < len(original_columns) else col_str))
                else:
                    seen_normalized[norm] = 0
                    new_cols.append(col_str)
            if duplicates_found:
                processed_df.columns = new_cols
                logger.warning(f"[Job {job_id}] Fixed {len(duplicates_found)} duplicate column names")
            logger.info(f"[Job {job_id}] Final columns: {processed_df.columns.tolist()}")
            logger.info(f"[Job {job_id}] Renamed {len(rename_map)} columns")
        
        # Step 5: Infer column types with LLM
        logger.info(f"[Job {job_id}] Inferring column types with LLM")
        column_types = llm_architect.infer_column_types(processed_df)
        logger.info(f"[Job {job_id}] Column types inferred: {len(column_types)} columns")
        
        # Step 6: Generate metadata with LLM
        logger.info(f"[Job {job_id}] Generating metadata with LLM")
        data_domain = metadata_generator.classify_domain(processed_df, table_name, analysis)
        comments = metadata_generator.generate_description(processed_df, table_name)
        period_col = metadata_generator.detect_period_column(processed_df)
        major_domain = metadata_generator.classify_major_domain(processed_df, table_name, analysis)
        sub_domain = metadata_generator.classify_sub_domain(processed_df, table_name, major_domain)
        
        llm_metadata = {
            'suggested_domain': major_domain,
            'description': comments,
            'period_column': period_col,
            'data_domain': data_domain,
            'major_domain': major_domain,
            'sub_domain': sub_domain,
            'data_category': data_category,
        }
        logger.info(f"[Job {job_id}] Metadata generated - Domain: {data_domain}, Major: {major_domain}")
        
        # Step 7: Generate signature and search for similar tables
        logger.info(f"[Job {job_id}] Generating table signature")
        signature, embedding = signature_builder.build_signature_with_embedding(
            df=processed_df,
            table_name=table_name,
            column_types=column_types
        )
        logger.info(f"[Job {job_id}] Signature and embedding generated")
        
        # Step 8: Connect to Milvus and search for similar tables
        logger.info(f"[Job {job_id}] Searching for similar tables in Milvus")
        milvus_connected = False
        similar_tables = []
        
        try:
            # Connect to Milvus
            if milvus_manager.connect():
                # Ensure collection exists
                milvus_manager.create_collection()
                milvus_connected = True
                
                # Search for similar tables
                similar_tables = milvus_manager.search_similar(
                    embedding=embedding,
                    top_k=settings.similarity_top_k,
                    threshold=settings.similarity_threshold
                )
                
                logger.info(f"[Job {job_id}] Found {len(similar_tables)} similar tables")
            else:
                logger.warning(f"[Job {job_id}] Could not connect to Milvus, proceeding with OTL")
        except Exception as e:
            logger.error(f"[Job {job_id}] Milvus search failed: {str(e)}, proceeding with OTL")
        
        # ── Phase A complete: store intermediate results, return context for Phase B ──
        job_manager.set_preprocessing_results(
            job_id=job_id,
            processed_df=processed_df,
            table_name=table_name,
            column_types=column_types,
            summary=summary,
            llm_metadata=llm_metadata,
        )
        logger.info(f"[Job {job_id}] Phase A complete — processed_df stored.")

        return {
            "df": df,
            "processed_df": processed_df,
            "table_name": table_name,
            "table_name_generated_by_llm": table_name_generated_by_llm,
            "column_types": column_types,
            "analysis": analysis,
            "summary": summary,
            "llm_metadata": llm_metadata,
            "signature": signature,
            "embedding": embedding,
            "similar_tables": similar_tables,
            "milvus_connected": milvus_connected,
            "normalized_sql_mode": normalized_sql_mode,
        }

    except Exception as e:
        logger.error(f"[Job {job_id}] Phase A failed: {str(e)}", exc_info=True)
        job_manager.set_error(job_id, f"Preprocessing failed: {str(e)}")
        return None


async def _preprocess_phase_b(
    job_id: str,
    file_path: str,
    file_description: Optional[str],
    ctx: dict,
):
    """Phase B: sequential IL resolution under lock."""
    df = ctx["df"]
    processed_df = ctx["processed_df"]
    table_name = ctx["table_name"]
    table_name_generated_by_llm = ctx["table_name_generated_by_llm"]
    column_types = ctx["column_types"]
    analysis = ctx["analysis"]
    llm_metadata = ctx["llm_metadata"]
    signature = ctx["signature"]
    embedding = ctx["embedding"]
    similar_tables = list(ctx["similar_tables"])  # copy to mutate locally
    milvus_connected = ctx["milvus_connected"]
    normalized_sql_mode = ctx["normalized_sql_mode"]
    summary = ctx["summary"]

    table_name_override = None
    if normalized_sql_mode in ("otl", "inc"):
        job = job_manager.get_job(job_id)
        if job:
            table_name_override = getattr(job, "proposed_table_name", None) or table_name

    try:
        # Step 9: Decide between OTL and IL
        matched_table_name = ""
        validation_result = {}
        report = ""
        is_compatible = False
        is_additive_evolution = False
        manual_inc_target = normalized_sql_mode == "inc" and bool(table_name_override)
        manual_otl_target = normalized_sql_mode == "otl" and bool(table_name_override)

        manual_inc_handled = False

        if manual_otl_target:
            # Explicit OTL requested — bypass all similarity matching and force
            # a fresh One-Time Load into the specified table name.
            logger.info(
                f"[Job {job_id}] Manual OTL mode selected. "
                f"Bypassing similarity matching and forcing new-table OTL for '{table_name_override}'."
            )
            similar_tables = []  # discard any Milvus / fallback matches
            # Also mark as handled so the column-based and peer-job fallback
            # paths in the else-branch below are skipped entirely.
            manual_inc_handled = True
        if manual_inc_target:
            target_table = str(table_name_override).strip().lower().replace(" ", "_")
            logger.info(
                f"[Job {job_id}] Manual INC mode selected. Forcing incremental target table: {target_table}"
            )

            new_columns = processed_df.columns.tolist()
            existing_db_types = db_manager.get_table_column_types(target_table)
            if not existing_db_types:
                raise ValueError(
                    f"Manual INC requested for '{target_table}', but target table does not exist."
                )

            new_types_lower = {k.lower(): v for k, v in column_types.items()}
            is_compatible, validation_result, report = schema_validator.validate_incremental_load(
                table_name=target_table,
                new_columns=new_columns,
                new_file_name=Path(file_path).name,
                new_types=new_types_lower,
                existing_types=existing_db_types,
            )
            is_additive_evolution = validation_result.get("is_additive_evolution", False)
            logger.info(
                f"[Job {job_id}] Manual INC schema validation: compatible={is_compatible}, "
                f"additive_evolution={is_additive_evolution}"
            )
            logger.info(f"[Job {job_id}] Manual INC validation report:\n{report}")

            similar_tables = [{
                "table_name": target_table,
                "similarity_score": 1.0,
                "created_at": "",
                "match_source": "manual_inc",
            }]

            job_manager.set_similarity_results(
                job_id=job_id,
                similar_tables=similar_tables,
                matched_table_name=target_table,
            )
            job_manager.set_schema_validation(
                job_id=job_id,
                schema_validation=validation_result,
                is_incremental_load=True,
            )

            logger.info(f"[Job {job_id}] Checking for duplicate data")
            duplicate_result = schema_validator.detect_duplicate_data(
                table_name=target_table,
                new_df=processed_df,
            )
            logger.info(f"[Job {job_id}] Duplicate detection: {duplicate_result['status']}")
            logger.info(f"[Job {job_id}] {duplicate_result['message']}")

            job = job_manager.get_job(job_id)
            if job:
                job.duplicate_detection = duplicate_result

            has_schema_changes = not is_compatible
            is_full_duplicate = duplicate_result["status"] == "DUPLICATE"

            if is_full_duplicate and not has_schema_changes:
                job_manager.update_status(job_id, JobStatus.DUPLICATE_DATA_DETECTED)
                logger.warning(
                    f"[Job {job_id}] Exact duplicate detected for manual INC target '{target_table}'. "
                    "Nothing new to load. Awaiting user decision."
                )
            else:
                job_manager.update_status(job_id, JobStatus.SCHEMA_MISMATCH)
                parts = []
                if duplicate_result["status"] in ["DUPLICATE", "PARTIAL_OVERLAP"]:
                    parts.append("duplicate/overlap detected (UPSERT will handle)")
                if is_compatible:
                    parts.append("exact schema match")
                elif is_additive_evolution:
                    parts.append("additive schema evolution — new columns will be added automatically")
                else:
                    parts.append("schema differences require review")
                logger.info(
                    f"[Job {job_id}] Manual INC queued for table '{target_table}': {'; '.join(parts)}."
                )
            manual_inc_handled = True

        elif similar_tables and len(similar_tables) > 0:
            # Take the top match
            top_match = similar_tables[0]
            matched_table_name = top_match['table_name']
            similarity_score = top_match['similarity_score']

            logger.info(f"[Job {job_id}] Top match: {matched_table_name} (similarity: {similarity_score:.2%})")

            # Validate schema compatibility
            # Also fetch actual DB column types for type-drift detection
            logger.info(f"[Job {job_id}] Validating schema compatibility")
            new_columns = processed_df.columns.tolist()

            # Fetch existing table column types for richer comparison
            existing_db_types = db_manager.get_table_column_types(matched_table_name)
            new_types_lower = {k.lower(): v for k, v in column_types.items()}

            is_compatible, validation_result, report = schema_validator.validate_incremental_load(
                table_name=matched_table_name,
                new_columns=new_columns,
                new_file_name=Path(file_path).name,
                new_types=new_types_lower,
                existing_types=existing_db_types
            )

            is_additive_evolution = validation_result.get('is_additive_evolution', False)

            # Guard: reject the match if column overlap is too low
            match_pct = validation_result.get('match_percentage', 0.0)
            if match_pct < settings.schema_match_min_percentage:
                logger.warning(
                    f"[Job {job_id}] Schema overlap too low ({match_pct:.1f}% < "
                    f"{settings.schema_match_min_percentage}%) for match '{matched_table_name}' "
                    f"— treating as new table (OTL)"
                )
                similar_tables = []  # discard match — fall through to OTL path below

            # Step 9b: LLM semantic verification — confirm the match is meaningful
            if similar_tables:
                logger.info(f"[Job {job_id}] Running LLM semantic verification for match '{matched_table_name}'")
                matched_metadata = schema_validator.fetch_table_metadata(matched_table_name) or {}
                semantic_result = schema_validator.verify_semantic_match(
                    matched_table_name=matched_table_name,
                    matched_table_metadata=matched_metadata,
                    new_table_name=table_name,
                    new_columns=new_columns,
                    new_llm_metadata=llm_metadata,
                    similarity_score=similarity_score,
                )
                logger.info(
                    f"[Job {job_id}] Semantic verification: is_related={semantic_result['is_related']}, "
                    f"confidence={semantic_result['confidence']:.2f}, "
                    f"reasoning='{semantic_result['reasoning']}'"
                )
                if not semantic_result['is_related']:
                    logger.warning(
                        f"[Job {job_id}] LLM rejected match '{matched_table_name}' as semantically "
                        f"unrelated (confidence={semantic_result['confidence']:.2f}) — treating as OTL"
                    )
                    similar_tables = []  # discard match — fall through to OTL path below

        # Re-check after potential guard rejection
        if manual_inc_handled:
            pass
        elif similar_tables and len(similar_tables) > 0:
            logger.info(
                f"[Job {job_id}] Schema validation: compatible={is_compatible}, "
                f"additive_evolution={is_additive_evolution}"
            )
            logger.info(f"[Job {job_id}] Validation report:\n{report}")

            # Store similarity and validation results
            job_manager.set_similarity_results(
                job_id=job_id,
                similar_tables=similar_tables,
                matched_table_name=matched_table_name
            )

            job_manager.set_schema_validation(
                job_id=job_id,
                schema_validation=validation_result,
                is_incremental_load=True
            )

            # Step 9.5: Detect duplicate data by comparing period values
            logger.info(f"[Job {job_id}] Checking for duplicate data")
            duplicate_result = schema_validator.detect_duplicate_data(
                table_name=matched_table_name,
                new_df=processed_df
            )

            logger.info(f"[Job {job_id}] Duplicate detection: {duplicate_result['status']}")
            logger.info(f"[Job {job_id}] {duplicate_result['message']}")

            # Store duplicate detection results
            job = job_manager.get_job(job_id)
            if job:
                job.duplicate_detection = duplicate_result

            # Determine final status
            #
            # UPSERT handles duplicates natively (ON CONFLICT DO UPDATE), so
            # duplicate data should NOT block the incremental load.  It is only
            # informational.  We still attach the duplicate_result on the job
            # so the user can see it, but we route to SCHEMA_MISMATCH (= IL
            # approval) in all cases except a pure exact re-upload with zero
            # schema changes — that one is flagged as DUPLICATE_DATA_DETECTED
            # so the user knows nothing new will happen.

            has_schema_changes = not is_compatible  # extra cols, missing cols, or type drift
            is_full_duplicate  = duplicate_result['status'] == 'DUPLICATE'

            if is_full_duplicate and not has_schema_changes:
                # Pure re-upload: identical data AND identical schema — warn user
                job_manager.update_status(job_id, JobStatus.DUPLICATE_DATA_DETECTED)
                logger.warning(
                    f"[Job {job_id}] Exact duplicate detected (same data + same schema). "
                    "Nothing new to load. Awaiting user decision."
                )
            else:
                # Route to IL approval — UPSERT will handle any overlapping rows,
                # and schema evolution will handle new/changed columns
                job_manager.update_status(job_id, JobStatus.SCHEMA_MISMATCH)

                # Build informational log message
                parts = []
                if duplicate_result['status'] in ['DUPLICATE', 'PARTIAL_OVERLAP']:
                    parts.append(f"duplicate/overlap detected (UPSERT will handle)")
                if is_compatible:
                    parts.append("exact schema match")
                elif is_additive_evolution:
                    parts.append("additive schema evolution — new columns will be added automatically")
                else:
                    parts.append("schema differences require review")

                logger.info(
                    f"[Job {job_id}] Incremental load queued: {'; '.join(parts)}."
                )
        else:
            # No Milvus match — try column-based fallback (IDF-weighted overlap)
            logger.info(f"[Job {job_id}] No Milvus match. Trying column-based fallback.")
            new_columns = processed_df.columns.tolist()
            fallback_result = schema_validator.find_similar_table_by_columns(
                new_columns=new_columns,
                min_overlap=settings.column_fallback_min_overlap,
            )

            if fallback_result:
                matched_table_name, overlap_score = fallback_result
                logger.info(
                    f"[Job {job_id}] Column fallback matched: {matched_table_name} "
                    f"(IDF-weighted overlap: {overlap_score:.2%})"
                )

                # Build a synthetic similar_tables entry so the rest of the
                # pipeline can reuse the same validation code path
                similarity_score = overlap_score
                similar_tables = [{
                    'table_name': matched_table_name,
                    'similarity_score': overlap_score,
                    'created_at': '',
                    'match_source': 'column_fallback',
                }]

                # -- Run the same validation pipeline as the Milvus path --

                # Fetch existing table column types for type-drift detection
                existing_db_types = db_manager.get_table_column_types(matched_table_name)
                new_types_lower = {k.lower(): v for k, v in column_types.items()}

                is_compatible, validation_result, report = schema_validator.validate_incremental_load(
                    table_name=matched_table_name,
                    new_columns=new_columns,
                    new_file_name=Path(file_path).name,
                    new_types=new_types_lower,
                    existing_types=existing_db_types
                )
                is_additive_evolution = validation_result.get('is_additive_evolution', False)

                # Guard: reject if column overlap is too low
                match_pct = validation_result.get('match_percentage', 0.0)
                if match_pct < settings.schema_match_min_percentage:
                    logger.warning(
                        f"[Job {job_id}] Column fallback schema overlap too low "
                        f"({match_pct:.1f}% < {settings.schema_match_min_percentage}%) "
                        f"for '{matched_table_name}' — treating as OTL"
                    )
                    similar_tables = []

                # LLM semantic verification
                if similar_tables:
                    logger.info(
                        f"[Job {job_id}] Running LLM semantic verification "
                        f"for column-fallback match '{matched_table_name}'"
                    )
                    matched_metadata = schema_validator.fetch_table_metadata(matched_table_name) or {}
                    semantic_result = schema_validator.verify_semantic_match(
                        matched_table_name=matched_table_name,
                        matched_table_metadata=matched_metadata,
                        new_table_name=table_name,
                        new_columns=new_columns,
                        new_llm_metadata=llm_metadata,
                        similarity_score=similarity_score,
                    )
                    logger.info(
                        f"[Job {job_id}] Semantic verification: "
                        f"is_related={semantic_result['is_related']}, "
                        f"confidence={semantic_result['confidence']:.2f}, "
                        f"reasoning='{semantic_result['reasoning']}'"
                    )
                    if not semantic_result['is_related']:
                        logger.warning(
                            f"[Job {job_id}] LLM rejected column-fallback match "
                            f"'{matched_table_name}' — treating as OTL"
                        )
                        similar_tables = []

                # If the fallback match survived all guards, route to IL
                if similar_tables:
                    logger.info(
                        f"[Job {job_id}] Column fallback validated. "
                        f"Schema: compatible={is_compatible}, "
                        f"additive_evolution={is_additive_evolution}"
                    )
                    logger.info(f"[Job {job_id}] Validation report:\n{report}")

                    job_manager.set_similarity_results(
                        job_id=job_id,
                        similar_tables=similar_tables,
                        matched_table_name=matched_table_name
                    )
                    job_manager.set_schema_validation(
                        job_id=job_id,
                        schema_validation=validation_result,
                        is_incremental_load=True
                    )

                    # Duplicate detection
                    logger.info(f"[Job {job_id}] Checking for duplicate data")
                    duplicate_result = schema_validator.detect_duplicate_data(
                        table_name=matched_table_name,
                        new_df=processed_df
                    )
                    logger.info(f"[Job {job_id}] Duplicate detection: {duplicate_result['status']}")
                    logger.info(f"[Job {job_id}] {duplicate_result['message']}")

                    job = job_manager.get_job(job_id)
                    if job:
                        job.duplicate_detection = duplicate_result

                    has_schema_changes = not is_compatible
                    is_full_duplicate = duplicate_result['status'] == 'DUPLICATE'

                    if is_full_duplicate and not has_schema_changes:
                        job_manager.update_status(job_id, JobStatus.DUPLICATE_DATA_DETECTED)
                        logger.warning(
                            f"[Job {job_id}] Exact duplicate detected via column fallback. "
                            "Awaiting user decision."
                        )
                    else:
                        job_manager.update_status(job_id, JobStatus.SCHEMA_MISMATCH)
                        parts = []
                        if duplicate_result['status'] in ['DUPLICATE', 'PARTIAL_OVERLAP']:
                            parts.append("duplicate/overlap detected (UPSERT will handle)")
                        if is_compatible:
                            parts.append("exact schema match")
                        elif is_additive_evolution:
                            parts.append("additive schema evolution — new columns will be added automatically")
                        else:
                            parts.append("schema differences require review")
                        logger.info(
                            f"[Job {job_id}] Incremental load queued (column fallback): "
                            f"{'; '.join(parts)}."
                        )
                else:
                    logger.info(
                        f"[Job {job_id}] Column fallback match rejected by guards. "
                        "Trying peer-job matching."
                    )

            # -- Peer-job matching (batch detection) --
            # If we still have no match, check other preprocessed jobs in memory
            if not similar_tables:
                logger.info(f"[Job {job_id}] Trying peer-job match (batch detection).")
                new_columns = processed_df.columns.tolist()
                peer_result = job_manager.find_peer_job_by_columns(
                    current_job_id=job_id,
                    new_columns=new_columns,
                    min_overlap=settings.column_fallback_min_overlap,
                )

                if peer_result:
                    peer_jid, peer_table, peer_score = peer_result
                    logger.info(
                        f"[Job {job_id}] Peer-job match found: job={peer_jid}, "
                        f"table={peer_table}, overlap={peer_score:.2%}"
                    )

                    # Build synthetic match entry
                    matched_table_name = peer_table
                    similarity_score = peer_score
                    similar_tables = [{
                        'table_name': peer_table,
                        'similarity_score': peer_score,
                        'created_at': '',
                        'match_source': 'peer_job',
                        'peer_job_id': peer_jid,
                    }]

                    # LLM semantic verification for the peer match
                    logger.info(
                        f"[Job {job_id}] Running LLM semantic verification "
                        f"for peer-job match '{peer_table}'"
                    )
                    peer_job = job_manager.get_job(peer_jid)
                    peer_metadata = peer_job.llm_metadata or {} if peer_job else {}
                    # Build a metadata dict compatible with verify_semantic_match
                    peer_meta_for_verify = {
                        'description': peer_metadata.get('description', ''),
                        'data_domain': peer_metadata.get('data_domain', ''),
                    }
                    peer_columns = (peer_job.processed_df.columns.tolist()
                                    if peer_job and peer_job.processed_df is not None
                                    else [])
                    semantic_result = schema_validator.verify_semantic_match(
                        matched_table_name=peer_table,
                        matched_table_metadata=peer_meta_for_verify,
                        new_table_name=table_name,
                        new_columns=new_columns,
                        new_llm_metadata=llm_metadata,
                        similarity_score=similarity_score,
                    )
                    logger.info(
                        f"[Job {job_id}] Peer-job semantic verification: "
                        f"is_related={semantic_result['is_related']}, "
                        f"confidence={semantic_result['confidence']:.2f}, "
                        f"reasoning='{semantic_result['reasoning']}'"
                    )
                    if not semantic_result['is_related']:
                        logger.warning(
                            f"[Job {job_id}] LLM rejected peer-job match "
                            f"'{peer_table}' (job {peer_jid}) -- treating as OTL"
                        )
                        similar_tables = []

                    # If peer match survived, route to IL
                    if similar_tables:
                        # Store the peer anchor reference
                        job = job_manager.get_job(job_id)
                        if job:
                            job.peer_anchor_job_id = peer_jid

                        job_manager.set_similarity_results(
                            job_id=job_id,
                            similar_tables=similar_tables,
                            matched_table_name=matched_table_name
                        )

                        # For peer matches, we do a simplified schema validation
                        # since the target table doesn't exist in DB yet
                        peer_cols = peer_columns
                        overlap_cols = set(c.lower() for c in new_columns) & set(c.lower() for c in peer_cols)
                        match_pct = (len(overlap_cols) / len(peer_cols) * 100) if peer_cols else 0
                        validation_result = {
                            'is_compatible': set(c.lower() for c in new_columns) == set(c.lower() for c in peer_cols),
                            'match_percentage': match_pct,
                            'matching_columns': list(overlap_cols),
                            'missing_columns': [c for c in peer_cols if c.lower() not in {x.lower() for x in new_columns}],
                            'extra_columns': [c for c in new_columns if c.lower() not in {x.lower() for x in peer_cols}],
                            'is_additive_evolution': len([c for c in new_columns if c.lower() not in {x.lower() for x in peer_cols}]) > 0
                                                     and len([c for c in peer_cols if c.lower() not in {x.lower() for x in new_columns}]) == 0,
                            'match_source': 'peer_job',
                        }

                        job_manager.set_schema_validation(
                            job_id=job_id,
                            schema_validation=validation_result,
                            is_incremental_load=True
                        )

                        job_manager.update_status(job_id, JobStatus.SCHEMA_MISMATCH)
                        logger.info(
                            f"[Job {job_id}] Peer-job incremental load detected. "
                            f"Target table: {peer_table} (from job {peer_jid}). "
                            f"Schema match: {match_pct:.1f}%"
                        )
                    else:
                        logger.info(
                            f"[Job {job_id}] Peer-job match rejected. "
                            "Proceeding with OTL."
                        )
                else:
                    logger.info(
                        f"[Job {job_id}] No matches found (Milvus + column fallback + peer jobs). "
                        "Proceeding with One-Time Load (OTL)."
                    )

        # Non-incremental safeguard: if LLM generated a table name that already
        # exists in DB, generate a unique OTL name to avoid accidental collision.
        job = job_manager.get_job(job_id)
        il_statuses = {JobStatus.SCHEMA_MISMATCH, JobStatus.DUPLICATE_DATA_DETECTED}
        is_incremental_path = job.status in il_statuses if job else False
        if table_name_generated_by_llm and not is_incremental_path and db_manager.table_exists(table_name):
            original_table_name = table_name
            table_name = _generate_non_colliding_table_name_via_llm(
                df=df,
                analysis=analysis,
                file_description=file_description,
                filename_stem=Path(file_path).stem,
                existing_name=table_name,
            )
            logger.warning(
                f"[Job {job_id}] LLM generated existing table name '{original_table_name}' "
                f"for non-incremental flow. Renamed to '{table_name}'."
            )

        # Step 10: Update table name if changed by collision safeguard, set final status
        job_manager.set_preprocessing_results(
            job_id=job_id,
            processed_df=processed_df,
            table_name=table_name,
            column_types=column_types,
            summary=summary,
            llm_metadata=llm_metadata
        )

        # Update status to AWAITING_APPROVAL only if not already set to IL/duplicate status
        job = job_manager.get_job(job_id)
        il_statuses = {JobStatus.SCHEMA_MISMATCH, JobStatus.DUPLICATE_DATA_DETECTED}
        if job.status not in il_statuses:
            job_manager.update_status(job_id, JobStatus.AWAITING_APPROVAL)
            logger.info(f"[Job {job_id}] Preprocessing complete. Awaiting user approval for OTL.")

        # Disconnect from Milvus
        if milvus_connected:
            milvus_manager.disconnect()

        logger.info(f"[Job {job_id}] Phase B complete. Pipeline finished.")
        
    except Exception as e:
        logger.error(f"[Job {job_id}] Phase B failed: {str(e)}", exc_info=True)
        job_manager.set_error(job_id, f"Preprocessing failed: {str(e)}")


async def insert_to_database(job_id: str, table_name: str, user_metadata):
    """
    Background task to insert processed data into PostgreSQL and save metadata.
    
    Args:
        job_id: Job identifier
        table_name: Name of the table to create/insert into
        user_metadata: User-provided metadata from approval request
    """
    logger.info(f"[Job {job_id}] Starting database insertion pipeline")
    
    try:
        job = job_manager.get_job(job_id)
        if not job or job.processed_df is None:
            raise ValueError("Job data not found or missing processed DataFrame")
        
        # Step 1: Save processed data as CSV
        logger.info(f"[Job {job_id}] Saving processed data as CSV")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"{timestamp}_{table_name}.csv"
        csv_path = PROCESSED_DIR / csv_filename
        job.processed_df.to_csv(csv_path, index=False)
        logger.info(f"[Job {job_id}] Saved to: {csv_path}")
        
        # Step 2: Create table
        logger.info(f"[Job {job_id}] Creating table: {table_name}")
        success = db_manager.create_table(table_name, job.column_types)
        if not success:
            raise Exception("Failed to create table")
        logger.info(f"[Job {job_id}] Table created successfully")
        
        # Step 3: Insert data
        logger.info(f"[Job {job_id}] Inserting data into table")
        rows_inserted = db_manager.insert_data(table_name, job.processed_df, column_types=job.column_types)
        logger.info(f"[Job {job_id}] Inserted {rows_inserted} rows")
        
        # Step 4: Insert metadata into tables_metadata
        logger.info(f"[Job {job_id}] Inserting metadata into tables_metadata")
        llm_meta = job.llm_metadata or {}
        period_col = llm_meta.get('period_column')
        
        # Generate period SQL queries if period column exists
        min_period_sql, max_period_sql = None, None
        if period_col:
            min_period_sql, max_period_sql = metadata_generator.generate_period_sql(table_name, period_col)
        
        tables_metadata_dict = {
            'data_domain': llm_meta.get('data_domain'),
            'table_name': table_name,
            'columns': ', '.join(job.processed_df.columns.tolist()),
            'comments': llm_meta.get('description'),
            'source': user_metadata.source,
            'source_url': user_metadata.source_url,
            'released_on': user_metadata.released_on,
            'updated_on': user_metadata.updated_on,
            'rows_count': rows_inserted,
            'business_metadata': user_metadata.business_metadata,
            'table_view': 'table',
            'period_cols': period_col,
            'min_period_sql': min_period_sql,
            'max_period_sql': max_period_sql
        }
        
        logger.info(f"[Job {job_id}] tables_metadata values:")
        logger.info(f"  - data_domain: {tables_metadata_dict['data_domain']}")
        logger.info(f"  - table_name: {tables_metadata_dict['table_name']}")
        logger.info(f"  - columns: {tables_metadata_dict['columns'][:100]}...")  # First 100 chars
        logger.info(f"  - comments: {tables_metadata_dict['comments']}")
        logger.info(f"  - source: {tables_metadata_dict['source']}")
        logger.info(f"  - source_url: {tables_metadata_dict['source_url']}")
        logger.info(f"  - released_on: {tables_metadata_dict['released_on']}")
        logger.info(f"  - updated_on: {tables_metadata_dict['updated_on']}")
        logger.info(f"  - rows_count: {tables_metadata_dict['rows_count']}")
        logger.info(f"  - period_cols: {tables_metadata_dict['period_cols']}")
        
        db_manager.insert_tables_metadata(tables_metadata_dict)
        logger.info(f"[Job {job_id}] ✓ tables_metadata record inserted successfully")
        
        # Step 5: Insert metadata into operational_metadata
        logger.info(f"[Job {job_id}] Inserting metadata into operational_metadata")
        
        # Get first and last period values
        first_val, last_val = None, None
        if period_col:
            first_val, last_val = metadata_generator.get_period_values(job.processed_df, period_col)
        
        operational_metadata_dict = {
            'table_name': table_name,
            'table_view': 'Table',
            'period_cols': period_col,
            'first_available_value': first_val,
            'last_available_value': last_val,
            'last_updated_on': datetime.now(),
            'rows_count': rows_inserted,
            'columns': ', '.join(job.processed_df.columns.tolist()),
            'source_url': user_metadata.source_url,
            'business_metadata': user_metadata.business_metadata,
            'major_domain': llm_meta.get('major_domain'),
            'sub_domain': llm_meta.get('sub_domain'),
            'brief_summary': llm_meta.get('description')
        }
        
        logger.info(f"[Job {job_id}] operational_metadata values:")
        logger.info(f"  - table_name: {operational_metadata_dict['table_name']}")
        logger.info(f"  - major_domain: {operational_metadata_dict['major_domain']}")
        logger.info(f"  - sub_domain: {operational_metadata_dict['sub_domain']}")
        logger.info(f"  - period_cols: {operational_metadata_dict['period_cols']}")
        logger.info(f"  - first_available_value: {operational_metadata_dict['first_available_value']}")
        logger.info(f"  - last_available_value: {operational_metadata_dict['last_available_value']}")
        logger.info(f"  - rows_count: {operational_metadata_dict['rows_count']}")
        logger.info(f"  - brief_summary: {operational_metadata_dict['brief_summary']}")
        
        db_manager.insert_operational_metadata(operational_metadata_dict)
        logger.info(f"[Job {job_id}] ✓ operational_metadata record inserted successfully")
        
        # Step 6: Store signature in Milvus for future similarity searches
        logger.info(f"[Job {job_id}] Storing table signature in Milvus")
        try:
            # Generate signature and embedding
            signature, embedding = signature_builder.build_signature_with_embedding(
                df=job.processed_df,
                table_name=table_name,
                column_types=job.column_types
            )
            
            # Connect to Milvus and store signature
            if milvus_manager.connect():
                milvus_manager.create_collection()
                success = milvus_manager.insert_signature(
                    table_name=table_name,
                    embedding=embedding,
                    signature=signature
                )
                
                if success:
                    logger.info(f"[Job {job_id}] ✓ Signature stored in Milvus")
                else:
                    logger.warning(f"[Job {job_id}] Failed to store signature in Milvus")
                
                milvus_manager.disconnect()
            else:
                logger.warning(f"[Job {job_id}] Could not connect to Milvus, signature not stored")
        except Exception as e:
            logger.error(f"[Job {job_id}] Error storing signature in Milvus: {str(e)}")
            # Don't fail the entire job if Milvus storage fails
        
        # Step 7: Update job with completion results
        job_manager.set_completion_results(
            job_id=job_id,
            final_table_name=table_name,
            rows_inserted=rows_inserted,
            processed_file_path=str(csv_path),
            warnings=[]
        )
        job_manager.update_status(job_id, JobStatus.COMPLETED)
        
        logger.info(f"[Job {job_id}] Database insertion and metadata tracking complete!")
        
    except Exception as e:
        logger.error(f"[Job {job_id}] Database insertion failed: {str(e)}", exc_info=True)
        job_manager.set_error(job_id, f"Database insertion failed: {str(e)}")


async def perform_incremental_load(job_id: str, table_name: str, user_metadata):
    """
    Background task: incremental load with schema evolution and UPSERT.

    This task:
    - Auto-detects primary keys from DB constraints
    - Evolves the table schema (ADD COLUMN for new fields, safe type widenings)
    - Performs deterministic UPSERT (INSERT ON CONFLICT DO UPDATE)
    - Maintains the ingested_at audit column
    - Returns a rich audit summary
    """
    logger.info(f"[Job {job_id}] Starting incremental load pipeline for table: {table_name}")
    
    try:
        job = job_manager.get_job(job_id)
        if not job or job.processed_df is None:
            raise ValueError("Job data not found or missing processed DataFrame")

        # Step 1: Perform incremental load with schema evolution + upsert
        logger.info(
            f"[Job {job_id}] Running schema-evolving UPSERT for "
            f"{len(job.processed_df)} rows into '{table_name}'"
        )

        load_summary = incremental_loader.perform_incremental_load(
            table_name=table_name,
            df=job.processed_df,
            column_types=job.column_types,
            key_columns=None,  # auto-detect from DB PK constraints
            last_available_value=None
        )

        if not load_summary['success']:
            raise Exception(f"Incremental load failed: {load_summary.get('error', 'Unknown error')}")

        rows_inserted = load_summary.get('rows_inserted', 0)
        rows_updated  = load_summary.get('rows_updated', 0)
        columns_added = load_summary.get('columns_added', [])
        schema_changes = load_summary.get('schema_changes', [])
        il_warnings    = load_summary.get('warnings', [])

        logger.info(f"[Job {job_id}] Incremental load summary:")
        logger.info(f"  - Rows before:    {load_summary['rows_before']}")
        logger.info(f"  - Rows inserted:  {rows_inserted}")
        logger.info(f"  - Rows updated:   {rows_updated}")
        logger.info(f"  - Rows after:     {load_summary['rows_after']}")
        if columns_added:
            logger.info(f"  - Columns added:  {columns_added}")
        if schema_changes:
            logger.info(f"  - Schema changes: {schema_changes}")
        if il_warnings:
            for w in il_warnings:
                logger.warning(f"  [W] {w}")

        # Step 2: Save processed data as CSV for record-keeping
        logger.info(f"[Job {job_id}] Saving incremental data as CSV")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"{timestamp}_{table_name}_incremental.csv"
        csv_path = PROCESSED_DIR / csv_filename
        job.processed_df.to_csv(csv_path, index=False)
        logger.info(f"[Job {job_id}] Saved to: {csv_path}")

        # Step 3: Update job with completion results
        job_manager.set_completion_results(
            job_id=job_id,
            final_table_name=table_name,
            rows_inserted=rows_inserted,
            processed_file_path=str(csv_path),
            warnings=il_warnings
        )
        # Store extra audit fields on the job object directly
        completed_job = job_manager.get_job(job_id)
        if completed_job:
            completed_job.rows_updated   = rows_updated
            completed_job.columns_added  = columns_added
            completed_job.schema_changes = schema_changes

        job_manager.update_status(job_id, JobStatus.INCREMENTAL_LOAD_COMPLETED)

        logger.info(
            f"[Job {job_id}] Incremental load complete! "
            f"+{rows_inserted} inserted / +{rows_updated} updated"
        )

    except Exception as e:
        logger.error(f"[Job {job_id}] Incremental load failed: {str(e)}", exc_info=True)
        job_manager.set_error(job_id, f"Incremental load failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
