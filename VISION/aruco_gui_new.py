import cv2
import cv2.aruco as aruco
import numpy as np
import time
import json
import math
import os
import csv
import threading
import tkinter as tk
from tkinter import messagebox
import sys
from collections import deque

temp_values = [deque(maxlen=200), deque(maxlen=200)]

# ====== Global Variables ======
LAPTOP_ID = 1
cap = None
running = False
tracker = None
MAX_TEMP_VALUES = 50

# array size 3 of calibrate base values
calibrate_values_x = [0] * 6
calibrate_values_y = [0] * 6
calibrate_values_z = [0] * 6

tracker_calibrated_x = [0]*6
tracker_calibrated_y = [0]*6
tracker_calibrated_z = [0]*6

last_sign = [0] * 2
# temp_values = [[], []] 

peak_data = []

peak_value = [0] * 2
save_lock = threading.Lock()

# Adding thread safety for tracker
tracker_lock = threading.Lock()

# ====== Marker & Tracker Classes ======
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
        m.x, m.y, m.z = tvec.flatten() * 1000  # m -> mm
        m.detected = True
        m.last_seen = time.time()

        # corners bisa shape (1,4,2) atau (4,2) atau array-like
        arr = np.array(corners)
        if arr.ndim == 3:   # (1,4,2)
            pts = arr[0]
        elif arr.ndim == 2: # (4,2)
            pts = arr
        else:
            pts = arr.reshape(-1, 2)
        m.last_pos_2d = tuple(np.mean(pts, axis=0).astype(int))


    def mark_all_undetected(self):
        for m in self.markers.values():
            m.detected = False

    def apply_timeout(self):
        now = time.time()
        for m in self.markers.values():
            if not m.detected and (now - m.last_seen > self.timeout):
                m.reset()

# ====== Camera Setup ======
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


# ====== Load Calibration ======
cameraMatrix = np.loadtxt('./calibration_2/mtx.csv', delimiter=',')
distCoeffs = np.loadtxt('./calibration_2/dist.csv', delimiter=',').reshape(-1, 1)

# ====== Detection Thread ======
def detection_loop():
    global cap, running, tracker, LAPTOP_ID  # removed undefined names

    global tracker_calibrated_x, tracker_calibrated_y, tracker_calibrated_z

    print("Starting detection...")
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    tracker = MarkerTracker(timeout=0.2)

    smoothed_fps = 0
    prev_time = time.perf_counter()
    running = True

    while running:
        ret, frame = cap.read()
        if not ret:
            cap = camera_connect(1)
            continue

        # calculate fps
        current_time = time.perf_counter()
        elapsed_time = current_time - prev_time
        prev_time = current_time
        smoothed_fps = (smoothed_fps * 0.9) + (1.0 / elapsed_time * 0.1)

        cv2.putText(frame, f"FPS: {smoothed_fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # beri kesempatan tracker untuk reset detection flag atomically
        with tracker_lock:
            tracker.mark_all_undetected()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            try:
                aruco.drawDetectedMarkers(frame, corners, ids)
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.08, cameraMatrix, distCoeffs)
                for i in range(len(ids)):
                    marker_id = int(ids[i][0])
                    with tracker_lock:
                        tracker.update_marker(marker_id, corners[i], rvecs[i], tvecs[i])
                    cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.03)
            except Exception as e:
                print(f"Error processing markers: {e}")

        for m in tracker.markers.values():
            if not m.detected:
                continue
            cx, cy = m.last_pos_2d
            text = (f"ID:{m.id} X:{m.x:.0f} Y:{m.y:.0f} Z:{m.z:.0f} "
                    f"R:{m.roll:.0f} P:{m.pitch:.0f} Y:{m.yaw:.0f}")
            cv2.putText(frame, text, (cx - 60, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


        tracker.apply_timeout()

        if (tracker[1].detected or tracker[2].detected or tracker[3].detected) and (tracker[4].detected or tracker[5].detected):
            x_base = y_base = z_base = 0.0

            for i in range(1, 6):
                if tracker[i].detected:
                    real_index = i + ((LAPTOP_ID - 1) * 5)
                    tracker_calibrated_x[i] = tracker[real_index].x - calibrate_values_x[i]
                    tracker_calibrated_y[i] = tracker[real_index].y - calibrate_values_y[i]
                    tracker_calibrated_z[i] = tracker[real_index].z - calibrate_values_z[i]
                
            if tracker[1].detected:
                x_base = tracker_calibrated_x[1]
                y_base = tracker_calibrated_y[1]
                z_base = tracker_calibrated_z[1]
                
            
            dx_top = tracker_calibrated_x[5] - x_base
            dy_top = tracker_calibrated_y[5] - y_base
            dz_top = tracker_calibrated_z[5] - z_base

            dx_low = tracker_calibrated_x[4] - x_base
            dy_low = tracker_calibrated_y[4] - y_base
            dz_low = tracker_calibrated_z[4] - z_base

            if mode_var.get() == "MODE_X":
                distance_top = dx_top
                distance_low = dx_low
            elif mode_var.get() == "MODE_X_Y":
                distance_top = math.sqrt(dx_top**2 + dy_top**2)
                distance_low = math.sqrt(dx_low**2 + dy_low**2)
            elif mode_var.get() == "MODE_Y":
                distance_top = dy_top
                distance_low = dy_low
            elif mode_var.get() == "MODE_X_Y_Z":
                distance_top = math.sqrt(dx_top**2 + dy_top**2 + dz_top**2)
                distance_low = math.sqrt(dx_low**2 + dy_low**2 + dz_low**2)

            # === Deteksi arah & peak ===
            global last_sign, temp_values, peak_data, peak_value

            sign = [None] * 2

            distances = [distance_top, distance_low]
            for i in range(2):
                sign[i] = 1 if distances[i] > 0 else -1
                temp_values[i].append(distances[i])
                # print(f"{len(temp_values[i])}")

                if (sign[i] != last_sign[i] and len(temp_values[i]) > 2 or len(temp_values[i]) >= (MAX_TEMP_VALUES-1)):
                    # Ambil peak dari siklus sebelumnya
                    if last_sign[i] != 0:  # bukan awal
                        if last_sign[i] > 0:
                            peak_value[i] = max(list(temp_values[i])[:-2])
                        elif last_sign[i] < 0:
                            peak_value[i] = min(list(temp_values[i])[:-2])

                        marker_name = "top marker" if i == 0 else "bottom marker"
                        print(f"Peak = {peak_value[i]:.2f} mm ({marker_name})")

                    # Reset untuk siklus berikutnya
                    last_sign[i] = sign[i]
                    temp_values[i].clear()

                    # Data gabungan status marker dan displacement
                    if i == 1:
                        data_entry = {
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "laptop_id": LAPTOP_ID,
                            "is_a_detected": tracker[5].detected,
                            "is_b_detected": tracker[4].detected,
                            "displacement_a": float(peak_value[0]),
                            "displacement_b": float(peak_value[1])
                        }
                        peak_data.append(data_entry)

                        # Simpan ke file
                        with save_lock:
                            with open("displacement.json", "w") as jf:
                                json.dump(data_entry, jf, indent=4)

                            file_exists = os.path.isfile("displacement_log.csv")
                            with open("displacement_log.csv", "a", newline="") as cf:
                                writer = csv.DictWriter(cf, fieldnames=data_entry.keys())
                                if not file_exists:
                                    writer.writeheader()
                                writer.writerow(data_entry)

        if peak_value[0] is not None:
            cv2.putText(frame, f"peak ={peak_value[0]:.2f}mm", (50, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if peak_value[1] is not None:
            cv2.putText(frame, f"peak ={peak_value[1]:.2f}mm", (50, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


        cv2.imshow("Marker Detection", frame)
        if cv2.waitKey(1) == 27 or not running:
            running = False
            break

    cap.release()
    cv2.destroyAllWindows()

# ====== GUI ======
def start_gui():
    global entry_id, label_status
    global mode_var

    root = tk.Tk()
    root.title("Marker Control Panel")
    root.geometry("320x420")

    tk.Label(root, text="Laptop ID:").pack(pady=5)
    entry_id = tk.Entry(root, width=10)
    entry_id.insert(0, "1")
    entry_id.pack(pady=5)

    # add mode selection
    tk.Label(root, text="Select Mode:").pack(pady=5)
    mode_var = tk.StringVar(value="MODE_X_Y_Z")
    modes = [("MODE_X", "MODE_X"), ("MODE_X_Y", "MODE_X_Y"), ("MODE_Y", "MODE_Y"), ("MODE_X_Y_Z", "MODE_X_Y_Z")]
    for text, mode in modes:
        tk.Radiobutton(root, text=text, variable=mode_var, value=mode).pack(anchor=tk.W)

    btn_start = tk.Button(root, text="Start Detection", bg="lightblue", command=start_detection)
    btn_start.pack(pady=5)

    btn_calibrate = tk.Button(root, text="Calibrate", bg="lightgreen", command=do_calibrate)
    btn_calibrate.pack(pady=5)

    btn_exit = tk.Button(root, text="Exit", bg="tomato", command=exit_program)
    btn_exit.pack(pady=5)

    label_status = tk.Label(root, text="Status: Idle", fg="blue")
    label_status.pack(pady=10)

    root.mainloop()


def start_detection():
    global LAPTOP_ID
    try:
        LAPTOP_ID = int(entry_id.get())
    except ValueError:
        messagebox.showwarning("Input Error", "Laptop ID harus berupa angka!")
        return

    thread = threading.Thread(target=detection_loop, daemon=True)
    thread.start()
    label_status.config(text=f"Status: Detection Running (ID={LAPTOP_ID})", fg="green")


def do_calibrate():
    global tracker

    if tracker is None:
        messagebox.showwarning("Not Ready", "Detection belum dimulai!")
        return

    time.sleep(0.3)

    BASE_MARKER = (LAPTOP_ID - 1) * 5
    with tracker_lock:
        for i in range(1,6):
            real_index = i + BASE_MARKER
            if tracker[i].detected:
                calibrate_values_x[i] = tracker[real_index].x
                calibrate_values_y[i] = tracker[real_index].y
                calibrate_values_z[i] = tracker[real_index].z

        status_text = (
            f"Status: Calibrated markers -\n"
            f"1: {'Yes' if tracker[BASE_MARKER + 1].detected else 'No'}\n"
            f"2: {'Yes' if tracker[BASE_MARKER + 2].detected else 'No'}\n"
            f"3: {'Yes' if tracker[BASE_MARKER + 3].detected else 'No'}\n"
            f"4: {'Yes' if tracker[BASE_MARKER + 4].detected else 'No'}\n"
            f"5: {'Yes' if tracker[BASE_MARKER + 5].detected else 'No'}"
        )

    label_status.config(text=status_text, fg="orange")
    messagebox.showinfo("Calibration", "Kalibrasi selesai!")

def exit_program():
    global running
    running = False
    time.sleep(0.1)
    if cap is not None:
        cap.release()
    time.sleep(0.1)
    cv2.destroyAllWindows()
    sys.exit()

if __name__ == "__main__":
    start_gui()
