from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from typing import Optional
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
    file_description: str = None
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
    logger.info(f"Received file upload: {file.filename}")
    
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
        background_tasks.add_task(preprocess_file, job_id, str(file_path), file_description)
        
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


# Background task functions

async def preprocess_file(job_id: str, file_path: str, file_description: str = None):
    """
    Background task to preprocess the uploaded file.
    
    Args:
        job_id: Job identifier
        file_path: Path to uploaded file
        file_description: Optional user-provided description to help with table naming
    """
    logger.info(f"[Job {job_id}] Starting preprocessing pipeline")
    
    try:
        # Step 1: Smart header detection
        logger.info(f"[Job {job_id}] Loading file for header detection: {file_path}")
        
        # Load first 5 rows without assuming header structure
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

            df_preview = pd.read_csv(
                file_path, header=None, nrows=5, sep=detected_delimiter, skiprows=skip_rows
            )
        else:
            df_preview = pd.read_excel(file_path, header=None, nrows=5)
        
        # Ask LLM to detect header row count
        header_count = llm_architect.detect_header_rows(df_preview)
        logger.info(f"[Job {job_id}] Detected {header_count} header row(s)")
        
        # Re-load file with correct header parameter
        if header_count == 1:
            # Single header row
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, sep=detected_delimiter, skiprows=skip_rows)
            else:
                df = pd.read_excel(file_path)
        elif header_count == 2:
            # Two header rows
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, header=[0, 1], sep=detected_delimiter, skiprows=skip_rows)
            else:
                df = pd.read_excel(file_path, header=[0, 1])
        else:  # header_count == 3
            # Three header rows
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, header=[0, 1, 2], sep=detected_delimiter, skiprows=skip_rows)
            else:
                df = pd.read_excel(file_path, header=[0, 1, 2])
        
        logger.info(f"[Job {job_id}] Loaded {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Step 2: Analyze file structure with LLM
        logger.info(f"[Job {job_id}] Analyzing file structure with LLM")
        analysis = llm_architect.analyze_file_structure(df)
        logger.info(f"[Job {job_id}] Analysis complete: {analysis}")
        
        # Step 2b: Classify as Category 1 or 2 (row-structure complexity)
        classification = category_classifier.classify(df, analysis)
        data_category = classification.get("category", 1)
        
        # Step 3: Generate table name with LLM
        logger.info(f"[Job {job_id}] Generating table name with LLM")
        table_name = llm_architect.generate_table_name(df, analysis, file_description)
        
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

        # Drop redundant/metadata columns first
        if drop_columns:
            drop_present = [c for c in drop_columns if c in processed_df.columns]
            if drop_present:
                processed_df = processed_df.drop(columns=drop_present)
                logger.info(f"[Job {job_id}] Dropped redundant/metadata columns: {drop_present}")

        # Apply cleaned column names (mapping only includes kept columns)
        if column_mapping:
            # Only rename columns that still exist
            rename_map = {k: v for k, v in column_mapping.items() if k in processed_df.columns}
            processed_df.rename(columns=rename_map, inplace=True)
            logger.info(f"[Job {job_id}] Columns after LLM cleaning: {processed_df.columns.tolist()}")

            # Ensure unique column names (handle any remaining duplicates from LLM)
            cols = processed_df.columns.tolist()
            seen = {}
            new_cols = []
            duplicates_found = []
            for idx, col in enumerate(cols):
                if col in seen:
                    seen[col] += 1
                    suffix = str(seen[col])
                    new_name = f"{col}_{suffix}"
                    new_cols.append(new_name)
                    duplicates_found.append((col, new_name, original_columns[idx] if idx < len(original_columns) else col))
                else:
                    seen[col] = 0
                    new_cols.append(col)
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
        
        # Step 9: Decide between OTL and IL
        if similar_tables and len(similar_tables) > 0:
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
            # No similar tables found, proceed with OTL
            logger.info(f"[Job {job_id}] No similar tables found. Proceeding with One-Time Load (OTL).")

        # Step 10: Store preprocessing results
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

        logger.info(f"[Job {job_id}] Preprocessing pipeline complete.")
        
    except Exception as e:
        logger.error(f"[Job {job_id}] Preprocessing failed: {str(e)}", exc_info=True)
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
