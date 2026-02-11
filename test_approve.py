"""
Quick diagnostic script to test the approval endpoint
Run this while the server is running to see what's happening
"""

import requests

# Test data
job_id = "test-job-id"  # Replace with actual job ID from browser console
url = f"http://localhost:8000/approve/{job_id}"

data = {
    'table_name': 'test_table',
    'source': 'Test Source',
    'source_url': 'https://example.com',
    'released_on': '2026-02-04T00:00:00',
    'updated_on': '2026-02-04T00:00:00',
    'business_metadata': 'Test metadata'
}

print(f"Testing POST to: {url}")
print(f"Data: {data}")
print("-" * 50)

try:
    response = requests.post(url, data=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
