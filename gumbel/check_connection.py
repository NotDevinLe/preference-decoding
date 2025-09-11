import requests

# Replace with the node where your server is running
NODE = "g3097"
PORT = 8002

BASE_URL = f"http://{NODE}:{PORT}"

def check_health():
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=10)
        print("Status:", r.status_code)
        print("Response:", r.text)
    except requests.exceptions.RequestException as e:
        print("Error:", e)


if __name__ == "__main__":
    check_health()
