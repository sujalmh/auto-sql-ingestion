import uuid
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List
from enum import Enum
import threading
import pandas as pd
from app.core.logger import logger


class JobStatus(str, Enum):
    """Job processing status enumeration."""
    PREPROCESSING = "preprocessing"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    COMPLETED = "completed"
    FAILED = "failed"
    # Incremental Load statuses
    SIMILARITY_SEARCH = "similarity_search"
    INCREMENTAL_LOAD_AUTO = "incremental_load_auto"
    SCHEMA_MISMATCH = "schema_mismatch"
    INCREMENTAL_LOAD_COMPLETED = "incremental_load_completed"
    DUPLICATE_DATA_DETECTED = "duplicate_data_detected"


class JobData:
    """Data structure for tracking job state."""
    
    def __init__(self, job_id: str, file_path: str):
        self.job_id = job_id
        self.status = JobStatus.PREPROCESSING
        self.file_path = file_path
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        
        # Preprocessing results
        self.processed_df: Optional[pd.DataFrame] = None
        self.proposed_table_name: Optional[str] = None
        self.column_types: Optional[Dict[str, str]] = None
        self.preprocessing_summary: Optional[str] = None
        self.llm_metadata: Optional[Dict] = None  # Store LLM-generated metadata
        self.data_category: Optional[int] = None  # 1 = standard, 2 = hierarchical/row-structure
        
        # Incremental Load fields
        self.similar_tables: Optional[List[Dict]] = None  # Similarity search results
        self.matched_table_name: Optional[str] = None  # Selected table for IL
        self.schema_validation: Optional[Dict] = None  # Schema validation result
        self.is_incremental_load: bool = False  # Flag for IL vs OTL
        self.duplicate_detection: Optional[Dict] = None  # Duplicate detection results
        
        # Final results
        self.final_table_name: Optional[str] = None
        self.rows_inserted: Optional[int] = None
        self.processed_file_path: Optional[str] = None
        self.warnings: list[str] = []
        
        # Error tracking
        self.error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert job data to dictionary (excluding DataFrame)."""
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "file_path": self.file_path,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "proposed_table_name": self.proposed_table_name,
            "final_table_name": self.final_table_name,
            "rows_inserted": self.rows_inserted,
            "processed_file_path": self.processed_file_path,
            "warnings": self.warnings,
            "error": self.error
        }


class JobManager:
    """
    Thread-safe job state manager.
    Stores job information in memory (can be upgraded to Redis for production).
    """
    
    def __init__(self, timeout_minutes: int = 30):
        self._jobs: Dict[str, JobData] = {}
        self._lock = threading.Lock()
        self.timeout_minutes = timeout_minutes
        logger.info(f"JobManager initialized with {timeout_minutes} minute timeout")
    
    def create_job(self, file_path: str) -> str:
        """
        Create a new job and return its ID.
        
        Args:
            file_path: Path to the uploaded file
            
        Returns:
            Generated job ID
        """
        job_id = str(uuid.uuid4())
        
        with self._lock:
            self._jobs[job_id] = JobData(job_id, file_path)
        
        logger.info(f"Created job {job_id} for file: {file_path}")
        return job_id
    
    def get_job(self, job_id: str) -> Optional[JobData]:
        """
        Retrieve job data by ID.
        
        Args:
            job_id: Job identifier
            
        Returns:
            JobData if found, None otherwise
        """
        with self._lock:
            return self._jobs.get(job_id)

    def clear_all(self) -> int:
        """Clear all jobs from memory. Returns number of jobs cleared."""
        with self._lock:
            n = len(self._jobs)
            self._jobs.clear()
        logger.info(f"Cleared {n} job(s) from memory")
        return n

    def update_status(
        self,
        job_id: str,
        status: JobStatus,
        error: Optional[str] = None,
        user_metadata: Optional[Dict] = None,
        table_name: Optional[str] = None
    ) -> bool:
        """
        Update job status.
        
        Args:
            job_id: Job identifier
            status: New status
            error: Error message if failed
            user_metadata: User-provided metadata
            table_name: Optional custom table name override
            
        Returns:
            True if successful, False if job not found
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = status
                job.updated_at = datetime.now()
                
                if error:
                    job.error = error
                
                if user_metadata:
                    job.user_metadata = user_metadata
                
                if table_name:
                    job.table_name = table_name
                
                logger.info(f"Job {job_id} status updated to: {status.value}")
                return True
            return False
    
    def set_preprocessing_results(
        self,
        job_id: str,
        processed_df: pd.DataFrame,
        table_name: str,
        column_types: Dict[str, str],
        summary: str,
        llm_metadata: Optional[Dict] = None
    ) -> bool:
        """
        Store preprocessing results.
        
        Args:
            job_id: Job identifier
            processed_df: Preprocessed DataFrame
            table_name: Proposed table name
            column_types: Inferred column types
            summary: Preprocessing summary
            llm_metadata: LLM-generated metadata (optional)
            
        Returns:
            True if successful, False if job not found
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.processed_df = processed_df
                job.proposed_table_name = table_name
                job.column_types = column_types
                job.preprocessing_summary = summary
                job.llm_metadata = llm_metadata
                job.data_category = llm_metadata.get("data_category") if llm_metadata else None
                job.updated_at = datetime.now()
                logger.info(f"Job {job_id} preprocessing results stored")
                return True
            return False
    
    def set_completion_results(
        self,
        job_id: str,
        final_table_name: str,
        rows_inserted: int,
        processed_file_path: str,
        warnings: list[str] = None
    ) -> bool:
        """
        Store completion results.
        
        Args:
            job_id: Job identifier
            final_table_name: Final table name used
            rows_inserted: Number of rows inserted
            processed_file_path: Path to saved CSV file
            warnings: Optional list of warnings
            
        Returns:
            True if successful, False if job not found
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.final_table_name = final_table_name
                job.rows_inserted = rows_inserted
                job.processed_file_path = processed_file_path
                job.warnings = warnings or []
                job.updated_at = datetime.now()
                logger.info(f"Job {job_id} completion results stored")
                return True
            return False
    
    def set_similarity_results(
        self,
        job_id: str,
        similar_tables: List[Dict],
        matched_table_name: Optional[str] = None
    ) -> bool:
        """
        Store similarity search results.
        
        Args:
            job_id: Job identifier
            similar_tables: List of similar table matches
            matched_table_name: Name of the matched table (if auto-selected)
            
        Returns:
            True if successful, False if job not found
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.similar_tables = similar_tables
                job.matched_table_name = matched_table_name
                job.updated_at = datetime.now()
                logger.info(f"Job {job_id} similarity results stored: {len(similar_tables)} matches")
                return True
            return False
    
    def set_schema_validation(
        self,
        job_id: str,
        schema_validation: Dict,
        is_incremental_load: bool = True
    ) -> bool:
        """
        Store schema validation results.
        
        Args:
            job_id: Job identifier
            schema_validation: Schema validation result dictionary
            is_incremental_load: Flag indicating if this is an IL job
            
        Returns:
            True if successful, False if job not found
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.schema_validation = schema_validation
                job.is_incremental_load = is_incremental_load
                job.updated_at = datetime.now()
                logger.info(f"Job {job_id} schema validation stored: compatible={schema_validation.get('is_compatible')}")
                return True
            return False
    
    def set_error(self, job_id: str, error: str) -> bool:
        """
        Set job error and update status to FAILED.
        
        Args:
            job_id: Job identifier
            error: Error message
            
        Returns:
            True if successful, False if job not found
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.error = error
                job.status = JobStatus.FAILED
                job.updated_at = datetime.now()
                logger.error(f"Job {job_id} failed: {error}")
                return True
            return False
    
    def is_expired(self, job_id: str) -> bool:
        """
        Check if a job has exceeded the approval timeout.
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if expired, False otherwise
        """
        job = self.get_job(job_id)
        if not job or job.status != JobStatus.AWAITING_APPROVAL:
            return False
        
        timeout_delta = timedelta(minutes=self.timeout_minutes)
        return datetime.now() - job.updated_at > timeout_delta
    
    def cleanup_expired_jobs(self) -> int:
        """
        Remove expired jobs from the manager.
        
        Returns:
            Number of jobs cleaned up
        """
        with self._lock:
            expired_ids = [
                job_id for job_id, job in self._jobs.items()
                if job.status == JobStatus.AWAITING_APPROVAL
                and datetime.now() - job.updated_at > timedelta(minutes=self.timeout_minutes)
            ]
            
            for job_id in expired_ids:
                del self._jobs[job_id]
            
            if expired_ids:
                logger.info(f"Cleaned up {len(expired_ids)} expired jobs")
            
            return len(expired_ids)


# Global job manager instance
job_manager = JobManager()
