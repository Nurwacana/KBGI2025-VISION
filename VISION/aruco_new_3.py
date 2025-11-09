import cv2
import cv2.aruco as aruco
import numpy as np
import time
import json
import math
import os
import csv

LAPTOP_ID = 1

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
        self.detected = False


class OffsetMarker:
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
        "dista": float(o.distance)
    }


def camera_connect(idx):
    count = 0
    while True:
        count += 1
        print(f"Reconnecting camera... (attempt {count})")
        cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 60)
            print("✅ Camera connected successfully")
            return cap
        time.sleep(0.4)


# ==== Load kalibrasi kamera ====
cameraMatrix = np.loadtxt('./calibration_2/mtx.csv', delimiter=',')
distCoeffs = np.loadtxt('./calibration_2/dist.csv', delimiter=',').reshape(-1, 1)

# ==== Setup kamera ====
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60)

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

tracker = MarkerTracker(timeout=0.05)

offset_1_2 = OffsetMarker(0)
offset_1_3 = OffsetMarker(1)

# Variabel peak tracking (global agar persist antar frame)
last_sign_a = 0
last_sign_b = 0
data_a = []
data_b = []
transition_counter_a = 0
transition_counter_b = 0
last_peak_a = 0.0
last_peak_b = 0.0
SIGN_TRANSITION_DELAY = 0

last_written_state = None
csv_filename = "displacement_log.csv"

count_peak = 0

prev_time = 0
smoothed_fps = 0.0
SMOOTHING = 0.9
first_time = 0
calibrate_topy_1 = 0.0
calibrate_topy_2 = 00
calibrate_topx_1 = 0.0
calibrate_topx_2 = 0.0

def main(LAPTOP_ID=LAPTOP_ID):
    global cap, last_sign_a, last_sign_b, data_a, data_b
    global transition_counter_a, transition_counter_b
    global last_peak_a, last_peak_b
    global count_peak
    global prev_time, smoothed_fps

    global first_time
    global calibrate_topy_1, calibrate_topy_2
    global calibrate_topx_1, calibrate_topx_2

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Frame not received, reconnecting camera...")
                cap = camera_connect(1)
                ret, frame = cap.read()
                continue

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

            # ==== Tampilkan data marker ====
            for m in tracker.markers.values():
                if not m.detected:
                    continue
                cx, cy = m.last_pos_2d
                text = (f"ID:{m.id} X:{m.x:.0f} Y:{m.y:.0f} Z:{m.z:.0f} "
                        f"R:{m.roll:.0f} P:{m.pitch:.0f} Y:{m.yaw:.0f}")
                cv2.putText(frame, text, (cx - 60, cy - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # ==== Hitung displacement antar marker ====
            if not first_time and tracker[1].detected and tracker[2].detected:
                calibrate_topy_1 = tracker[1].y
                calibrate_topx_1 = tracker[1].x
                calibrate_topy_2 = tracker[2].y
                calibrate_topx_2 = tracker[2].x
                first_time = 1
               

            # if tracker[1].detected and tracker[2].detected:
            #     dy_cal_1 = tracker[1].y - distance_calibrate_from_top_1
            #     dy_cal_2 = tracker[2].y - distance_calibrate_from_top_2
            #     dx = tracker[2].x - tracker[1].x
            #     dy = dy_cal_2 - dy_cal_1
            #     dz = tracker[2].z - tracker[1].z
            #     offset_1_2.distance = dy  # hanya sumbu X
            #     offset_1_2.detected = True
            #     cv2.line(frame, tracker[1].last_pos_2d, tracker[2].last_pos_2d, (255, 0, 0), 2)
            # else:
            #     offset_1_2.distance = 0.0
            #     offset_1_2.detected = False

            if tracker[1].detected and tracker[2].detected:
                # Hitung selisih posisi terhadap titik kalibrasi awal
                dx_1 = tracker[1].x - calibrate_topx_1
                dy_1 = tracker[1].y - calibrate_topy_1
                dx_2 = tracker[2].x - calibrate_topx_2
                dy_2 = tracker[2].y - calibrate_topy_2

                # Offset relatif antara marker 1 dan marker 2
                dx = dx_2 - dx_1
                dy = dy_2 - dy_1

                # Hitung besar getaran total (resultan)
                distance_2d = math.sqrt(dx**2 + dy**2)

                # Tentukan tanda berdasarkan kuadran
                if (dy >= 0 and dx >= 0) or (dy >= 0 and dx < 0):  # kuadran 1 atau 2
                    offset_1_2.distance = +distance_2d
                else:  # kuadran 3 atau 4
                    offset_1_2.distance = -distance_2d

                offset_1_2.detected = True
                cv2.line(frame, tracker[1].last_pos_2d, tracker[2].last_pos_2d, (255, 0, 0), 2)
            else:
                offset_1_2.distance = 0.0
                offset_1_2.detected = False


            if tracker[1].detected and tracker[3].detected:
                dx = tracker[3].x - tracker[1].x
                dy = tracker[3].y - tracker[1].y
                offset_1_3.distance = np.sqrt(dx**2 + dy**2)
                offset_1_3.detected = True
                cv2.line(frame, tracker[1].last_pos_2d, tracker[3].last_pos_2d, (0, 255, 0), 2)
            else:
                offset_1_3.distance = 0.0
                offset_1_3.detected = False

            displacement_a = offset_1_2.distance
            displacement_b = offset_1_3.distance

            # ==== Deteksi Peak ====
            def get_sign(x):
                if x > 0:
                    return 1
                elif x < 0:
                    return -1
                return 0

            # ===== Marker A =====
            sign_a = get_sign(displacement_a)
            if last_sign_a == 0 and sign_a != 0:
                last_sign_a = sign_a
                data_a = [displacement_a]
            elif sign_a == last_sign_a and sign_a != 0:
                if transition_counter_a > 0:
                    transition_counter_a -= 1
                else:
                    data_a.append(displacement_a)
            elif sign_a != last_sign_a and sign_a != 0:
                if data_a:
                    last_peak_a = max(data_a) if last_sign_a > 0 else min(data_a)
                    count_peak += 1
                    print(f"🔵 Peak A ({'+' if last_sign_a > 0 else '-'}) = {last_peak_a:.2f} mm {count_peak}")
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
                    last_peak_b = max(data_b) if last_sign_b > 0 else min(data_b)
                    print(f"🟢 Peak B ({'+' if last_sign_b > 0 else '-'}) = {last_peak_b:.2f} mm")
                data_b.clear()
                last_sign_b = sign_b
                transition_counter_b = SIGN_TRANSITION_DELAY

            # ==== Update state ke JSON ====
            state = {
                "laptop_id": LAPTOP_ID,
                "is_a_detected": offset_1_2.detected,
                "is_b_detected": offset_1_3.detected,
                "displacement_a": float(last_peak_a),
                "displacement_b": float(last_peak_b)
            }



            with open("displacement.json", "w") as f:
                json.dump(state, f, indent=4)

            global last_written_state

            # Cek apakah file CSV sudah ada, kalau belum buat header
            if not os.path.exists(csv_filename):
                with open(csv_filename, "w", newline="") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=state.keys())
                    writer.writeheader()

            # Bandingkan dengan state terakhir
            if last_written_state != state:
                # Simpan ke file CSV
                with open(csv_filename, "a", newline="") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=state.keys())
                    writer.writerow(state)
                last_written_state = state.copy()

            # print(state)

            # display fps smoothed
            now = time.perf_counter()
            dt = now - prev_time
            prev_time = now
            fps = 1 / dt if dt > 0 else 0
            smoothed_fps = SMOOTHING * smoothed_fps + (1 - SMOOTHING) * fps

            cv2.putText(frame, f"FPS: {smoothed_fps:.1f}", (10, 30),    
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.imshow("Marker Detection", frame)
            if cv2.waitKey(1) == 27:
                break

        except Exception as e:
            print(f"Error: {e}")
            time.sleep(0.2)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    input_id = input("Enter laptop ID (default 1): ")
    if input_id.isdigit():
        LAPTOP_ID = int(input_id)
    main(LAPTOP_ID)
