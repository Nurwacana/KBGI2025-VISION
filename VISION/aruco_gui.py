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

# ====== Global Variables ======
LAPTOP_ID = 1
cap = None
running = False
tracker = None

calibrate_topx_1 = calibrate_topy_1 = 0.0
calibrate_topx_2 = calibrate_topy_2 = 0.0

# array size 3 of calibrate base values
calibrate_basex = []
calibrate_basey = []

last_sign = 0
temp_values = []
peak_data = []
save_lock = threading.Lock()

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
        m.last_pos_2d = tuple(np.mean(corners[0], axis=0).astype(int))

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
    global cap, running, tracker, calibrate_topy_1, calibrate_topy_2
    global calibrate_topx_1, calibrate_topx_2, LAPTOP_ID

    print("Starting detection...")
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    tracker = MarkerTracker(timeout=0.05)

    smoothed_fps = 0
    prev_time = time.perf_counter()
    running = True

    while running:
        ret, frame = cap.read()
        if not ret:
            cap = camera_connect(1)
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

        for m in tracker.markers.values():
            if not m.detected:
                continue
            cx, cy = m.last_pos_2d
            text = (f"ID:{m.id} X:{m.x:.0f} Y:{m.y:.0f} Z:{m.z:.0f} "
                    f"R:{m.roll:.0f} P:{m.pitch:.0f} Y:{m.yaw:.0f}")
            cv2.putText(frame, text, (cx - 60, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


        tracker.apply_timeout()

        # Hitung offset 1-2
        # Hitung offset 1-2
        if tracker[1].detected and tracker[2].detected:
            dx_1 = tracker[1].x - calibrate_topx_1
            dy_1 = tracker[1].y - calibrate_topy_1
            dx_2 = tracker[2].x - calibrate_topx_2
            dy_2 = tracker[2].y - calibrate_topy_2

            dx = dx_2 - dx_1
            dy = dy_2 - dy_1

            if mode_var.get() == "MODE_X":
                distance_2d = dx
            elif mode_var.get() == "MODE_X_Y":
                distance_2d = math.sqrt(dx**2 + dy**2) * (1 if dy >= 0 else -1)
            elif mode_var.get() == "MODE_Y":
                distance_2d = dy

            # === Deteksi arah & peak ===
            global last_sign, temp_values, peak_data

            sign = 1 if distance_2d > 0 else -1
            temp_values.append(distance_2d)
            print(temp_values)

            # Cek pergantian arah (dengan jeda 2 frame untuk stabilitas)
            if sign != last_sign and len(temp_values) > 2:
                # Ambil peak dari siklus sebelumnya
                if last_sign != 0:  # bukan awal
                    if last_sign > 0:
                        peak_val = max(temp_values[:-2])
                        direction = "Up"
                    elif last_sign < 0:
                        peak_val = min(temp_values[:-2])
                        direction = "Down"

                    # Data gabungan status marker dan displacement
                    data_entry = {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "laptop_id": LAPTOP_ID,
                        "is_a_detected": tracker[1].detected,
                        "is_b_detected": tracker[2].detected,
                        "displacement_a": float(distance_2d),
                        "displacement_b": float(tracker[2].y - calibrate_topy_2)
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

                    print(f"Peak = {peak_val:.2f} mm [{direction}]")


                # Reset untuk siklus berikutnya
                temp_values = []
                last_sign = sign

            # Gambar garis dan nilai di frame
            cv2.line(frame, tracker[1].last_pos_2d, tracker[2].last_pos_2d, (255, 0, 0), 2)
            cv2.putText(frame, f"peak ={distance_2d:.2f}mm", (50, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            last_sign = 0
            temp_values = []

        cv2.imshow("Marker Detection", frame)
        if cv2.waitKey(1) == 27:
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
    root.geometry("320x360")

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
    global calibrate_topy_1, calibrate_topy_2, calibrate_topx_1, calibrate_topx_2
    global tracker

    if tracker is None:
        messagebox.showwarning("Not Ready", "Detection belum dimulai!")
        return

    time.sleep(0.3)
    if tracker[1].detected and tracker[2].detected:
        calibrate_topy_1 = tracker[1].y
        calibrate_topx_1 = tracker[1].x
        calibrate_topy_2 = tracker[2].y
        calibrate_topx_2 = tracker[2].x
        # for 1, 2
        for i in [1, 2]:
            calibrate_basex.append(tracker[i].x)
            calibrate_basey.append(tracker[i].y)
        label_status.config(text="Status: ✅ Calibrated", fg="green")
        messagebox.showinfo("Calibration", "Kalibrasi berhasil!")
    else:
        label_status.config(text="⚠️ Marker 1 & 2 belum terdeteksi", fg="red")
        messagebox.showwarning("Calibration Failed", "Pastikan Marker 1 & 2 terlihat kamera!")


def exit_program():
    global running
    running = False
    if  cap is not None:
        cap.release()
    time.sleep(0.1)
    cv2.destroyAllWindows()
    os._exit(0)


# ====== Run GUI ======
if __name__ == "__main__":
    start_gui()
