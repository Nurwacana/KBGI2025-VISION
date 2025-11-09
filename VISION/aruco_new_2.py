import cv2
import cv2.aruco as aruco
import numpy as np
import time
import json
import math
import os

LAPTOP_ID = 1

# ==== Struct mirip C++ ====
class Marker:
    def __init__(self, id_):
        self.id = id_
        self.detected = False
        self.x = self.y = self.z = 0.0
        self.roll = self.pitch = self.yaw = 0.0
        self.last_seen = 0.0
        self.last_pos_2d = (0, 0)

    def reset(self):
        self.x = self.y = self.z = 0.0
        self.roll = self.pitch = self.yaw = 0.0

class offsetMarker:
    def __init__(self, id_):
        self.id = id_
        self.detected = False
        self.distance = 0.0

class MarkerTracker:
    def __init__(self, timeout=1.0):
        self.markers = {}
        self.timeout = timeout

    def __getitem__(self, id_):
        if id_ not in self.markers:
            self.markers[id_] = Marker(id_)
        return self.markers[id_]

    def update_marker(self, id_, corners=None, rvec=None, tvec=None):
        m = self[id_]
        R, _ = cv2.Rodrigues(rvec)
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        m.roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        m.pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        m.yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        m.x, m.y, m.z = tvec.flatten() * 1000  # meter -> mm
        m.detected = True
        m.last_seen = time.time()
        m.last_pos_2d = tuple(np.mean(corners[0], axis=0).astype(int))

    def mark_all_undetected(self):
        for m in self.markers.values():
            m.detected = False

    def apply_timeout(self):
        now = time.time()
        for m in self.markers.values():
            if not m.detected and (now - m.last_seen > self.timeout):
                m.reset()

def offset_to_dict(o):
    return {
        "id": int(o.id),
        "is_dista_detected": bool(o.detected),
        "dista": float(o.distance),
        "is_disb_detected": bool(o,)
    }

def camera_connect(idx):
    count = 0
    while True:
        count+=1
        print(f"retrying {count}")
        cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
        ret, frame = cap.read()
        if ret:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 60)
            break
        time.sleep(400)

dista = 0.0
distb = 0.0

# ==== Load kalibrasi kamera ====
cameraMatrix = np.loadtxt('./calibration_2/mtx.csv', delimiter=',')
distCoeffs = np.loadtxt('./calibration_2/dist.csv', delimiter=',').reshape(-1, 1)

# ==== Setup kamera ====
cap = cv2.VideoCapture(1, cv2.CAP_MSMF)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60)

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

tracker = MarkerTracker(timeout=0.01)

offset_1_2 = offsetMarker(0)
offset_1_3 = offsetMarker(1)

is_a = 0
is_b = 0

displacement_a = 0.0
displacement_b = 0.0

# ==== Variabel untuk deteksi peak ====
prev_a = curr_a = next_a = 0.0
prev_b = curr_b = next_b = 0.0
last_peak_a = 0.0
last_peak_b = 0.0
min_interval = 0.0  # 50 ms antar peak (hindari noise)
last_peak_time_a = 0
last_peak_time_b = 0


while True:
    try: 
        ret, frame = cap.read()
        if not ret:
            camera_connect(1)
            # ret, frame = cap.read()

        tracker.mark_all_undetected()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.08, cameraMatrix, distCoeffs)

            for i in range(len(ids)):
                marker_id = int(ids[i][0])
                tracker.update_marker(marker_id, corners[i], rvec[i], tvec[i])
                cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvec[i], tvec[i], 0.03)

        tracker.apply_timeout()

        for m in tracker.markers.values():
            if not m.detected:
                continue
            cx, cy = m.last_pos_2d
            text = (f"ID:{m.id} "
                    f"X:{m.x:.0f} Y:{m.y:.0f} Z:{m.z:.0f} "
                    f"R:{m.roll:.0f} P:{m.pitch:.0f} Y:{m.yaw:.0f}")
            cv2.putText(frame, text, (cx - 60, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if tracker[1].detected and tracker[2].detected:
            dx = tracker[2].x - tracker[1].x
            dy = tracker[2].y - tracker[1].y
            dz = tracker[2].z - tracker[1].z
            dist = np.sqrt(dx**2 + dy**2 + dz**2)
            # offset_1_2.distance = dist
            offset_1_2.distance = dx
            # print(f"Distance between Marker 1 and 2: {dist:.1f} mm")
            cv2.line(frame, tracker[1].last_pos_2d, tracker[2].last_pos_2d, (255, 0, 0), 2)
        else:
            offset_1_2.distance = 0.0

        distance_from_camera = np.sqrt(tracker[1].x**2 + tracker[1].y**2 + tracker[1].z**2)
        # if tracker[1].detected:
        #     print(f"distance x 1: {tracker[1].x:.1f} mm, y 1: {tracker[1].y:.1f} mm, z 1: {tracker[1].z:.1f} mm, distance from camera: {distance_from_camera:.1f} mm")

        if tracker[1].detected and tracker[3].detected:
            dx = tracker[3].x - tracker[1].x
            dy = tracker[3].y - tracker[1].y
            dist = np.sqrt(dx**2 + dy**2)
            offset_1_3.distance = dist
            # print(f"Distance between Marker 1 and 3: {dist:.1f} mm")
            cv2.line(frame, tracker[1].last_pos_2d, tracker[3].last_pos_2d, (0, 255, 0), 2)
        else:
            offset_1_3.distance = 0.0

        is_a = offset_1_2.detected
        is_b = offset_1_3.detected

        displacement_a = offset_1_2.distance
        displacement_b = offset_1_3.distance

        # ==== Deteksi Peak dua arah dengan jeda 2 data untuk hindari noise ====
        SIGN_TRANSITION_DELAY = 2

        def get_sign(x):
            if x > 0:
                return 1
            elif x < 0:
                return -1
            return 0

        # Inisialisasi variabel saat pertama kali loop
        if 'last_sign_a' not in locals():
            last_sign_a = 0
            last_sign_b = 0
            data_a = []
            data_b = []
            transition_counter_a = 0
            transition_counter_b = 0

        # ===== Marker A =====
        sign_a = get_sign(displacement_a)

        # Jika arah belum ada dan mulai muncul arah baru
        if last_sign_a == 0 and sign_a != 0:
            last_sign_a = sign_a
            data_a = [displacement_a]

        # Masih di arah yang sama
        elif sign_a == last_sign_a and sign_a != 0:
            if transition_counter_a > 0:
                transition_counter_a -= 1
            else:
                data_a.append(displacement_a)

        # Jika arah berubah (misal dari + ke - atau sebaliknya)
        elif sign_a != last_sign_a and sign_a != 0:
            # Simpulkan peak dari arah sebelumnya
            if data_a:
                if last_sign_a > 0:
                    last_peak_a = max(data_a)
                else:
                    last_peak_a = min(data_a)
                print(f"🔵 Peak A arah {'positif' if last_sign_a > 0 else 'negatif'}: {last_peak_a:.2f} mm")

            # Reset buffer dan mulai jeda transisi
            data_a.clear()
            last_sign_a = sign_a
            transition_counter_a = SIGN_TRANSITION_DELAY

        # ===== Marker B =====
        sign_b = get_sign(displacement_b)

        if last_sign_b == 0 and sign_b != 0:
            last_sign_b = sign_b
            data_b = [displacement_b]

        elif sign_b == last_sign_b and sign_b != 0:
            if transition_counter_b > 0:
                transition_counter_b -= 1
            else:
                data_b.append(displacement_b)

        elif sign_b != last_sign_b and sign_b != 0:
            if data_b:
                if last_sign_b > 0:
                    last_peak_b = max(data_b)
                else:
                    last_peak_b = min(data_b)
                print(f"🟢 Peak B arah {'positif' if last_sign_b > 0 else 'negatif'}: {last_peak_b:.2f} mm")

            data_b.clear()
            last_sign_b = sign_b
            transition_counter_b = SIGN_TRANSITION_DELAY

        # ==== Update state tanpa mengubah strukturnya ====
        state = {
            "laptop_id": LAPTOP_ID,
            "is_a_detected": is_a,
            "is_b_detected": is_b,
            "displacement_a": last_peak_a,
            "displacement_b": last_peak_b
        }

        print(state)

        with open("displacement.json", "w") as f:
            json.dump(state, f, indent=4)





        cv2.imshow('Marker Detection', frame)
        if cv2.waitKey(1) == 27:
            break
    except Exception as e:
        print(f"Error: {e}")


cap.release()
cv2.destroyAllWindows()
