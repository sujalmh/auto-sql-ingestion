import sys
import requests
import json

def delete_job(job_id: str):
    url = f"http://localhost:8002/job/{job_id}"
    print(f"Sending DELETE request to {url}...")
    try:
        response = requests.delete(url)
        print(f"Status Code: {response.status_code}")
        print("Response JSON:")
        print(json.dumps(response.json(), indent=2))
    except requests.exceptions.ConnectionError:
        print("Failed to connect to the server. Is it running on http://localhost:8002?")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/delete_job.py <job_id>")
        sys.exit(1)
    
    target_job_id = sys.argv[1]
    delete_job(target_job_id)
