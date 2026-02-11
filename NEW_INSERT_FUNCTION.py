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
        
        # Step 6: Update job with completion results
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
