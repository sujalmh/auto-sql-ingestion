from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.responses import JSONResponse, FileResponse
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
            df_preview = pd.read_csv(file_path, header=None, nrows=5)
        else:
            df_preview = pd.read_excel(file_path, header=None, nrows=5)
        
        # Ask LLM to detect header row count
        header_count = llm_architect.detect_header_rows(df_preview)
        logger.info(f"[Job {job_id}] Detected {header_count} header row(s)")
        
        # Re-load file with correct header parameter
        if header_count == 1:
            # Single header row
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
        elif header_count == 2:
            # Two header rows
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, header=[0, 1])
            else:
                df = pd.read_excel(file_path, header=[0, 1])
        else:  # header_count == 3
            # Three header rows
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, header=[0, 1, 2])
            else:
                df = pd.read_excel(file_path, header=[0, 1, 2])
        
        logger.info(f"[Job {job_id}] Loaded {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Step 2: Analyze file structure with LLM
        logger.info(f"[Job {job_id}] Analyzing file structure with LLM")
        analysis = llm_architect.analyze_file_structure(df)
        logger.info(f"[Job {job_id}] Analysis complete: {analysis}")
        
        # Step 3: Generate table name with LLM
        logger.info(f"[Job {job_id}] Generating table name with LLM")
        table_name = llm_architect.generate_table_name(df, analysis, file_description)
        
        # Step 3b: Refine table name if too long
        table_name = llm_architect.refine_table_name(table_name)
        logger.info(f"[Job {job_id}] Final table name: {table_name}")
        
        # Step 4: Preprocess data
        logger.info(f"[Job {job_id}] Preprocessing data")
        processed_df, summary = preprocessor.preprocess(df, analysis)
        logger.info(f"[Job {job_id}] Preprocessing complete: {summary}")
        
        # Step 4b: Clean column names with LLM
        logger.info(f"[Job {job_id}] Cleaning column names")
        original_columns = processed_df.columns.tolist()
        logger.info(f"[Job {job_id}] Original columns before cleaning: {original_columns}")
        
        column_mapping = llm_architect.clean_column_names(original_columns)
        
        # Apply cleaned column names
        if column_mapping:
            processed_df.rename(columns=column_mapping, inplace=True)
            logger.info(f"[Job {job_id}] Columns after LLM cleaning: {processed_df.columns.tolist()}")
            
            # ALWAYS ensure unique column names (handle duplicates from LLM cleaning)
            cols = processed_df.columns.tolist()
            seen = {}
            new_cols = []
            duplicates_found = []
            
            for idx, col in enumerate(cols):
                if col in seen:
                    seen[col] += 1
                    # Try to extract meaningful suffix from original column name
                    orig_col = original_columns[idx]
                    # Note: `self._extract_suffix_from_original` is not defined in this scope.
                    # Assuming a simple counter-based suffix for now, similar to original logic.
                    # If a more sophisticated suffix extraction is needed, a helper function
                    # would need to be defined globally or passed in.
                    suffix = str(seen[col]) # Placeholder for actual suffix extraction logic
                    new_name = f"{col}_{suffix}"
                    new_cols.append(new_name)
                    duplicates_found.append((col, new_name, orig_col))
                else:
                    seen[col] = 0
                    new_cols.append(col)
            
            if duplicates_found:
                processed_df.columns = new_cols
                logger.warning(f"[Job {job_id}] Fixed {len(duplicates_found)} duplicate column names:")
                for old, new, orig in duplicates_found:
                    logger.warning(f"  '{old}' → '{new}' (from original: '{orig}')")
            
            logger.info(f"[Job {job_id}] Final columns: {processed_df.columns.tolist()}")
            logger.info(f"[Job {job_id}] Renamed {len(column_mapping)} columns")
        
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
            'sub_domain': sub_domain
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
            logger.info(f"[Job {job_id}] Validating schema compatibility")
            new_columns = processed_df.columns.tolist()
            
            is_compatible, validation_result, report = schema_validator.validate_incremental_load(
                table_name=matched_table_name,
                new_columns=new_columns,
                new_file_name=Path(file_path).name
            )
            
            logger.info(f"[Job {job_id}] Schema validation: compatible={is_compatible}")
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
            
            # Determine final status based on duplicate detection
            if duplicate_result['status'] in ['DUPLICATE', 'PARTIAL_OVERLAP']:
                # Warn user about duplicate/overlap
                job_manager.update_status(job_id, JobStatus.DUPLICATE_DATA_DETECTED)
                logger.warning(f"[Job {job_id}] Duplicate/overlap detected. Awaiting user decision.")
            else:
                # NEW_DATA or NO_PERIOD_COLUMN - proceed with normal IL approval
                job_manager.update_status(job_id, JobStatus.SCHEMA_MISMATCH)
                logger.info(f"[Job {job_id}] Incremental load detected. Awaiting user approval.")
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
        
        # Update status to AWAITING_APPROVAL if not already set to SCHEMA_MISMATCH
        job = job_manager.get_job(job_id)
        if job.status != JobStatus.SCHEMA_MISMATCH:
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
        rows_inserted = db_manager.insert_data(table_name, job.processed_df)
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
    Background task to perform incremental load (append data to existing table).
    
    Args:
        job_id: Job identifier
        table_name: Name of the existing table to append to
        user_metadata: User-provided metadata from approval request
    """
    logger.info(f"[Job {job_id}] Starting incremental load pipeline for table: {table_name}")
    
    try:
        job = job_manager.get_job(job_id)
        if not job or job.processed_df is None:
            raise ValueError("Job data not found or missing processed DataFrame")
        
        # Step 1: Perform incremental load
        logger.info(f"[Job {job_id}] Appending {len(job.processed_df)} rows to {table_name}")
        
        load_summary = incremental_loader.perform_incremental_load(
            table_name=table_name,
            df=job.processed_df,
            column_types=job.column_types,
            last_available_value=None  # Can be enhanced to track period values
        )
        
        if not load_summary['success']:
            raise Exception(f"Incremental load failed: {load_summary.get('error', 'Unknown error')}")
        
        logger.info(f"[Job {job_id}] Incremental load summary:")
        logger.info(f"  - Rows before: {load_summary['rows_before']}")
        logger.info(f"  - Rows added: {load_summary['rows_added']}")
        logger.info(f"  - Rows after: {load_summary['rows_after']}")
        
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
            rows_inserted=load_summary['rows_added'],
            processed_file_path=str(csv_path),
            warnings=[]
        )
        job_manager.update_status(job_id, JobStatus.INCREMENTAL_LOAD_COMPLETED)
        
        logger.info(f"[Job {job_id}] Incremental load complete!")
        
    except Exception as e:
        logger.error(f"[Job {job_id}] Incremental load failed: {str(e)}", exc_info=True)
        job_manager.set_error(job_id, f"Incremental load failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
