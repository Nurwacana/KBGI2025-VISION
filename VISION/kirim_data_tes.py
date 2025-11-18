import requests
import json
import time

# ======== KONFIGURASI ========
URL = "http://192.168.43.236/detector-getaran/api/receive_camera_data.php"
HEADERS = {"Content-Type": "application/json"}
LAPTOP_ID = 1  # ubah sesuai ID laptop 1–8
RETRIES = 3

# ======== FUNGSI KIRIM DATA ========
def send_dummy():
    data = {
        "laptop_id": LAPTOP_ID,
        "dista": 123.45,
        "distb": 67.89,
        "is_a_detected": True,
        "is_b_detected": True
    }

    json_data = json.dumps(data)
    session = requests.Session()

    for attempt in range(1, RETRIES + 1):
        try:
            response = session.post(URL, data=json_data, headers=HEADERS, timeout=(3, 5))
            print(f"\n✅ [OK] Attempt {attempt}: {response.status_code}")
            print("Response:", response.text)
            return True
        except requests.exceptions.RequestException as e:
            print(f"❌ [Error] Attempt {attempt}: {e}")
            time.sleep(1)

    print("🚫 Gagal mengirim data setelah beberapa percobaan.")
    return False

# ======== LOOP PENGUJIAN ========
if __name__ == "__main__":
    print("=== Tes Kirim Data Dummy ke Server ===")
    while True:
        send_dummy()
        time.sleep(1)  # jeda 1 detik antar kiriman
