import sys
import argparse
from datetime import datetime
from pathlib import Path
import psycopg2
from psycopg2 import sql

# Add the project root to sys.path so we can import app modules
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from app.config import settings
from app.core.milvus_manager import milvus_manager
from app.core.logger import logger

UPLOAD_DIR = project_root / "uploads"
PROCESSED_DIR = project_root / "processed"

def delete_by_date(target_date_str: str, dry_run: bool = False):
    try:
        target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
    except ValueError:
        logger.error("Invalid date format. Please use YYYY-MM-DD")
        sys.exit(1)

    logger.info(f"Starting deletion for date: {target_date} (Dry Run: {dry_run})")

    # 1. Connect to PostgreSQL to find tables ingested on this date
    conn = None
    tables_to_delete = []
    try:
        conn = psycopg2.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            dbname=settings.postgres_db,
            user=settings.postgres_user,
            password=settings.postgres_password
        )
        with conn.cursor() as cur:
            # Query tables_metadata
            cur.execute("""
                SELECT table_name 
                FROM tables_metadata 
                WHERE DATE(created_at) = %s
            """, (target_date,))
            
            rows = cur.fetchall()
            tables_to_delete = [row[0] for row in rows]
            
            logger.info(f"Found {len(tables_to_delete)} tables ingested on {target_date}: {tables_to_delete}")
            
            if not dry_run:
                for table_name in tables_to_delete:
                    # Drop the table
                    cur.execute(sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(sql.Identifier(table_name)))
                    
                    # Delete from tables_metadata
                    cur.execute("DELETE FROM tables_metadata WHERE table_name = %s", (table_name,))
                    
                    # Delete from operational_metadata
                    cur.execute("DELETE FROM operational_metadata WHERE table_name = %s", (table_name,))
                    
                    logger.info(f"Dropped table and cleared DB metadata for: {table_name}")
                
                conn.commit()
    except Exception as e:
        logger.error(f"Database error: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

    # 2. Delete from Milvus
    if tables_to_delete:
        try:
            if milvus_manager.connect():
                if dry_run:
                    logger.info(f"[Dry Run] Would delete Milvus signatures for: {tables_to_delete}")
                else:
                    for table_name in tables_to_delete:
                        if milvus_manager.delete_signature(table_name):
                            logger.info(f"Deleted Milvus signature for: {table_name}")
                milvus_manager.disconnect()
        except Exception as e:
            logger.error(f"Milvus error: {e}")

    # 3. Cleanup Files in uploads/ and processed/
    logger.info("Cleaning up files in uploads and processed directories...")
    files_deleted = 0
    
    # Format of timestamp in uploads: YYYYMMDD_HHMMSS
    target_date_prefix = target_date.strftime("%Y%m%d")
    
    # Uploads directory cleanup
    if UPLOAD_DIR.exists():
        for upload_file in UPLOAD_DIR.iterdir():
            if not upload_file.is_file():
                continue
                
            # Check if file name matches the date prefix OR modified time matches
            file_mtime_date = datetime.fromtimestamp(upload_file.stat().st_mtime).date()
            if upload_file.name.startswith(f"{target_date_prefix}_") or file_mtime_date == target_date:
                if dry_run:
                    logger.info(f"[Dry Run] Would delete upload file: {upload_file.name}")
                else:
                    upload_file.unlink()
                    logger.info(f"Deleted upload file: {upload_file.name}")
                    files_deleted += 1

    # Processed directory cleanup
    if PROCESSED_DIR.exists():
        for processed_file in PROCESSED_DIR.iterdir():
            if not processed_file.is_file():
                continue
            
            file_mtime_date = datetime.fromtimestamp(processed_file.stat().st_mtime).date()
            
            # Delete if modified on target date, OR if filename matches one of the tables we are deleting
            matches_table = any(f"_{t}" in processed_file.name for t in tables_to_delete)
            
            if file_mtime_date == target_date or matches_table:
                if dry_run:
                    logger.info(f"[Dry Run] Would delete processed file: {processed_file.name}")
                else:
                    processed_file.unlink()
                    logger.info(f"Deleted processed file: {processed_file.name}")
                    files_deleted += 1

    logger.info(f"Done! Cleaned up {len(tables_to_delete)} tables and {files_deleted} files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete ingested tables, metadata, and files by date.")
    parser.add_argument("--date", required=True, help="Target date in YYYY-MM-DD format")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be deleted without actually deleting")
    
    args = parser.parse_args()
    delete_by_date(args.date, args.dry_run)
