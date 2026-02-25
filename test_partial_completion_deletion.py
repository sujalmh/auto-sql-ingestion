import requests
import time
import os
import uuid
import psycopg2
from pathlib import Path

BASE_URL = "http://localhost:8011"

print("Starting partial completion deletion test...")

# Setup: Start the server manually or assume it's running.
# Let's test standard delete_job with a mock job_id directly to see if our globbing logic breaks.
dummy_job_id = str(uuid.uuid4())

# Create a dummy processed file directly to simulate a partial completion
os.makedirs("processed", exist_ok=True)
partial_csv = "processed/20260223_180000_partial_table_test.csv"
with open(partial_csv, "w") as f:
    f.write("a,b\n1,2")

# We want to test if our API cleans this up if a job had this table name.
# Unforunately, we need a real job in memory to hit the endpoint. So let's upload a file.
with open("test_partial.csv", "w") as f:
    f.write("id,val\n1,10\n")

print("1. Uploading file to get a real job_id...")
with open("test_partial.csv", "rb") as f:
    res = requests.post(f"{BASE_URL}/upload", files={"file": f})
job_id = res.json()["job_id"]
print(f"Uploaded. Job ID: {job_id}")

while True:
    res = requests.get(f"{BASE_URL}/status/{job_id}").json()
    if res["status"] in ["awaiting_approval", "failed", "completed"]:
        break
    time.sleep(1)

# Approve it but give it our partial table name.
# Then IMMEIDATELY delete it before it can finish insertion, simulating a partial failure or interruption!
# Or we can just let it complete, but test if the globbing finds extra files!
print("2. Approving job with custom table name 'partial_table_test'...")
approve_data = {
    "table_name": "partial_table_test",
    "source": "t", "source_url": "u", "released_on": "2024-01-01T00:00:00Z", "updated_on": "2024-01-01T00:00:00Z"
}
requests.post(f"{BASE_URL}/approve/{job_id}", data=approve_data)

# don't wait, delete immediately!
print("3. Deleting job immediately to catch partial state...")
try:
    del_res = requests.delete(f"{BASE_URL}/job/{job_id}")
    print(f"Delete response: {del_res.status_code} - {del_res.json()}")
except Exception as e:
    print(f"Delete failed: {e}")

# Check if our dummy file was deleted by the globbing mechanism!
if os.path.exists(partial_csv):
    print(f"FAIL: The dangling file {partial_csv} was NOT deleted!")
else:
    print(f"SUCCESS: The dangling file {partial_csv} WAS deleted by partial completion cleanup!")

# Cleanup our test file
if os.path.exists("test_partial.csv"):
    os.remove("test_partial.csv")
