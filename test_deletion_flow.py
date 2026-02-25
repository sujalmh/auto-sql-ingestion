import requests
import time
import os
import psycopg2
from pathlib import Path

BASE_URL = "http://localhost:8010"

# Create a dummy CSV file
csv_content = "id,name,value\n1,test1,100\n2,test2,200\n"
with open("test_ingestion.csv", "w") as f:
    f.write(csv_content)

print("1. Uploading file...")
with open("test_ingestion.csv", "rb") as f:
    files = {"file": ("test_ingestion.csv", f, "text/csv")}
    data = {"file_description": "Test Data for Deletion"}
    res = requests.post(f"{BASE_URL}/upload", files=files, data=data)
    
if res.status_code != 200:
    print(f"Failed to upload: {res.text}")
    exit(1)

job_id = res.json()["job_id"]
print(f"Uploaded successfully. Job ID: {job_id}")

print("2. Waiting for awaiting_approval status...")
status = "preprocessing"
while status == "preprocessing":
    time.sleep(2)
    res = requests.get(f"{BASE_URL}/status/{job_id}")
    status = res.json()["status"]
    print(f"Current status: {status}")

if status != "awaiting_approval":
    print(f"Unexpected status: {status}")
    if "error" in res.json():
        print(f"Error: {res.json()['error']}")
    exit(1)

proposed_table = res.json()["preview"]["proposed_table_name"]
table_name = "test_deletion_table_abc" # override table name
print(f"Proposed table: {proposed_table}, overriding to {table_name}")

print("3. Approving job...")
approve_data = {
    "table_name": table_name,
    "source": "Test Script",
    "source_url": "http://test",
    "released_on": "2024-01-01T00:00:00Z",
    "updated_on": "2024-01-01T00:00:00Z",
    "business_metadata": "Test Data"
}
res = requests.post(f"{BASE_URL}/approve/{job_id}", data=approve_data)
if res.status_code != 200:
    print(f"Failed to approve: {res.text}")
    exit(1)

print("4. Waiting for completion...")
status = "approved"
while status not in ["completed", "failed"]:
    time.sleep(2)
    res = requests.get(f"{BASE_URL}/status/{job_id}")
    status = res.json()["status"]
    print(f"Current status: {status}")

if status != "completed":
    print(f"Job failed: {res.json().get('error')}")
    exit(1)

print("Job completed successfully. Checking database to ensure table exists.")
# DB check
import sys
sys.path.append(os.getcwd())
from app.config import settings
conn = psycopg2.connect(
    host=settings.postgres_host,
    port=settings.postgres_port,
    database=settings.postgres_db,
    user=settings.postgres_user,
    password=settings.postgres_password
)
cur = conn.cursor()
cur.execute(f"SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = '{table_name}')")
exists = cur.fetchone()[0]
if not exists:
    print("Error: Table was not created!")
    exit(1)
else:
    print(f"Table {table_name} exists.")

print("5. Deleting job...")
res = requests.delete(f"{BASE_URL}/job/{job_id}")
print(f"Delete response: {res.status_code} - {res.json()}")

print("6. Verifying deletion...")
res = requests.get(f"{BASE_URL}/status/{job_id}")
if res.status_code == 404:
    print("Job GET returned 404 (Expected).")
else:
    print(f"Error: Job GET returned {res.status_code}")

cur.execute(f"SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = '{table_name}')")
exists = cur.fetchone()[0]
if exists:
    print(f"Error: Table {table_name} still exists!")
else:
    print(f"Success: Table {table_name} was deleted.")

cur.execute(f"SELECT EXISTS (SELECT FROM tables_metadata WHERE table_name = '{table_name}')")
meta_exists = cur.fetchone()[0]
if meta_exists:
    print("Error: tables_metadata still exists!")
else:
    print("Success: tables_metadata was deleted.")

conn.close()
os.remove("test_ingestion.csv")
print("All tests passed successfully!")
