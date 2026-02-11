

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


