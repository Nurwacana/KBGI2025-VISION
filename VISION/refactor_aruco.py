"""
refactor_aruco_baseid.py
Class-based refactor that preserves the original displacement and peak-detection logic
and enforces BASE_MARKER = (laptop_id - 1) * 5 semantics consistently.

- Does NOT change calculation logic or peak-detection logic you requested.
- Uses safe shutdown (Event) and proper numpy checks to avoid ValueError.
- Keeps synchronous file I/O for displacement (to preserve throughput you required).
- Does not impose any hard limit on marker IDs; it purely relies on laptop_id and markers
  observed by the camera. The user-specified laptop_id determines the BASE_MARKER.

Usage: python refactor_aruco_baseid.py
"""

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
import requests

# ====== Configuration ======
CAMERA_INDEX_DEFAULT = 1
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_FPS = 60
MARKER_SIZE_M = 0.08  # 8 cm
MAX_TEMP_VALUES = 50

# ====== Global-ish (kept similar names) ======
save_lock = threading.Lock()

# ====== Marker & Tracker Classes (same interface) ======
class Marker:
    def __init__(self, id_):
        self.id = int(id_)
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
        self.markers = {}  # dict of id -> Marker
        self.timeout = timeout
        self.lock = threading.Lock()

    def __getitem__(self, id_):
        # create marker if missing, preserving original semantics
        id_ = int(id_)
        if id_ not in self.markers:
            self.markers[id_] = Marker(id_)
        return self.markers[id_]

    def update_marker(self, id_, corners=None, rvec=None, tvec=None):
        m = self[id_]
        try:
            R, _ = cv2.Rodrigues(rvec)
            sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
            m.roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
            m.pitch = np.degrees(np.arctan2(-R[2, 0], sy))
            m.yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
            m.x, m.y, m.z = tvec.flatten() * 1000  # m -> mm
            m.detected = True
            m.last_seen = time.time()

            arr = np.array(corners)
            if arr.ndim == 3:
                pts = arr[0]
            elif arr.ndim == 2:
                pts = arr
            else:
                pts = arr.reshape(-1, 2)
            m.last_pos_2d = tuple(np.mean(pts, axis=0).astype(int))
        except Exception as e:
            # tolerant as original
            print(f"update_marker error id={id_}: {e}")

    def mark_all_undetected(self):
        for m in self.markers.values():
            m.detected = False

    def apply_timeout(self):
        now = time.time()
        for m in self.markers.values():
            if not m.detected and (now - m.last_seen > self.timeout):
                m.reset()


# ====== Camera handler as class but preserving logic ======
class CameraHandler(threading.Thread):
    def __init__(self, camera_index=CAMERA_INDEX_DEFAULT, laptop_id=1,
                 mtx_path='./calibration_2/mtx.csv', dist_path='./calibration_2/dist.csv'):
        super().__init__(daemon=True)
        self.camera_index = camera_index
        self.laptop_id = int(laptop_id)
        self.cap = None
        self.tracker = MarkerTracker(timeout=0.2)

        # stop event for safe shutdown
        self._stop_event = threading.Event()

        # load calibration
        try:
            self.cameraMatrix = np.loadtxt(mtx_path, delimiter=',')
            self.distCoeffs = np.loadtxt(dist_path, delimiter=',').reshape(-1, 1)
        except Exception as e:
            print(f"Failed to load calibration: {e}")
            self.cameraMatrix = None
            self.distCoeffs = None

        # calibration baseline arrays (1..5 indices used)
        self.calibrate_values_x = [0.0] * 6
        self.calibrate_values_y = [0.0] * 6
        self.calibrate_values_z = [0.0] * 6

        # tracker calibrated arrays (1..5)
        self.tracker_calibrated_x = [0.0] * 6
        self.tracker_calibrated_y = [0.0] * 6
        self.tracker_calibrated_z = [0.0] * 6

        # peak/temp like original
        self.temp_values = [deque(maxlen=200), deque(maxlen=200)]
        self.last_sign = [0, 0]
        self.peak_value = [None, None]

        self.x_base = 0.0
        self.y_base = 0.0
        self.z_base = 0.0

        self.prev_timer = [0.0, 0.0]

        self.calibrated = [False] * 6

        # GUI mode var reference (set by GUI)
        self.mode_var = None

    def stop(self):
        self._stop_event.set()
        try:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    def camera_connect(self, idx):
        count = 0
        while not self._stop_event.is_set():
            count += 1
            print(f"Reconnecting camera... (attempt {count})")
            try:
                cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                    cap.set(cv2.CAP_PROP_FPS, FRAME_FPS)
                    print("✅ Camera connected successfully")
                    return cap
                else:
                    try:
                        cap.release()
                    except Exception:
                        pass
            except Exception as e:
                print(f"camera_connect error: {e}")
            for _ in range(4):
                if self._stop_event.is_set():
                    break
                time.sleep(0.1)
        return None

    def run(self):
        # safe check for numpy arrays
        if self.cameraMatrix is None or self.distCoeffs is None or getattr(self.cameraMatrix, 'size', 0) == 0 or getattr(self.distCoeffs, 'size', 0) == 0:
            print("Camera calibration missing or invalid. Aborting camera thread.")
            return

        # open camera
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_MSMF)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, FRAME_FPS)

        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
        parameters = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(aruco_dict, parameters)

        smoothed_fps = 0
        prev_time = time.perf_counter()

        while not self._stop_event.is_set():
            ret, frame = (False, None)
            try:
                if self.cap is None:
                    self.cap = self.camera_connect(self.camera_index)
                    if self.cap is None:
                        break
                ret, frame = self.cap.read()
            except Exception:
                ret = False

            if not ret:
                try:
                    self.cap = self.camera_connect(self.camera_index)
                    if self.cap is None:
                        break
                except Exception:
                    time.sleep(0.05)
                    continue

            if frame is None:
                ret, frame = self.cap.read()
                continue

            # calculate fps
            current_time = time.perf_counter()
            elapsed_time = current_time - prev_time
            prev_time = current_time
            if elapsed_time > 0:
                smoothed_fps = (smoothed_fps * 0.9) + (1.0 / elapsed_time * 0.1)

            # overlay fps
            try:
                cv2.putText(frame, f"FPS: {smoothed_fps:.2f}", (10, FRAME_HEIGHT - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            except Exception:
                pass

            # mark all undetected (protected by lock)
            try:
                with self.tracker.lock:
                    self.tracker.mark_all_undetected()
            except Exception:
                self.tracker.mark_all_undetected()


            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)

            if ids is not None:
                try:
                    aruco.drawDetectedMarkers(frame, corners, ids)
                    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE_M, self.cameraMatrix, self.distCoeffs)
                    for i in range(len(ids)):
                        marker_id = int(ids[i][0])
                        try:
                            with self.tracker.lock:
                                self.tracker.update_marker(marker_id, corners[i], rvecs[i], tvecs[i])
                        except Exception:
                            self.tracker.update_marker(marker_id, corners[i], rvecs[i], tvecs[i])
                        cv2.drawFrameAxes(frame, self.cameraMatrix, self.distCoeffs, rvecs[i], tvecs[i], 0.03)
                except Exception as e:
                    print(f"Error processing markers: {e}")

            # draw marker info (kept minimal to avoid overhead)
            for m in list(self.tracker.markers.values()):
                if not m.detected:
                    continue
                cx, cy = m.last_pos_2d
                text = (f"ID:{m.id} X:{m.x:.0f} Y:{m.y:.0f} Z:{m.z:.0f} "
                        f"R:{m.roll:.0f} P:{m.pitch:.0f} Y:{m.yaw:.0f}")
                try:
                    cv2.putText(frame, text, (cx - 60, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                except Exception:
                    pass

            # apply timeout like original
            self.tracker.apply_timeout()

            # ----- CRITICAL: Preserve exact logic using BASE_MARKER -----
            base = (self.laptop_id - 1) * 5
            if (self.tracker[base + 1].detected or self.tracker[base + 2].detected or self.tracker[base + 3].detected):
                

                for i in range(1, 6):
                    real_index = base + i
                    if real_index in self.tracker.markers and self.tracker.markers[real_index].detected:
                        self.tracker_calibrated_x[i] = self.tracker.markers[real_index].x - self.calibrate_values_x[i]
                        self.tracker_calibrated_y[i] = self.tracker.markers[real_index].y - self.calibrate_values_y[i]
                        self.tracker_calibrated_z[i] = self.tracker.markers[real_index].z - self.calibrate_values_z[i]

                        if i == 4 or i == 5: 
                            self.prev_timer[5-i] = time.time()

                if self.tracker[base + 1].detected:
                    self.x_base = self.tracker_calibrated_x[1]
                    self.y_base = self.tracker_calibrated_y[1]
                    self.z_base = self.tracker_calibrated_z[1]
                elif self.tracker[base + 2].detected:
                    self.x_base = self.tracker_calibrated_x[2]
                    self.y_base = self.tracker_calibrated_y[2]
                    self.z_base = self.tracker_calibrated_z[2]
                elif self.tracker[base + 3].detected:
                    self.x_base = self.tracker_calibrated_x[3]
                    self.y_base = self.tracker_calibrated_y[3]
                    self.z_base = self.tracker_calibrated_z[3]

                if self.calibrated[5]:
                    dx_top = self.tracker_calibrated_x[5] - self.x_base
                    dy_top = self.tracker_calibrated_y[5] - self.y_base
                    dz_top = self.tracker_calibrated_z[5] - self.z_base
                else:
                    dx_top = dy_top = dz_top = 0.0

                if self.calibrated[4]:
                    dx_low = self.tracker_calibrated_x[4] - self.x_base
                    dy_low = self.tracker_calibrated_y[4] - self.y_base
                    dz_low = self.tracker_calibrated_z[4] - self.z_base
                else:
                    dx_low = dy_low = dz_low = 0.0

                try:
                    mode = self.mode_var.get()
                except Exception:
                    mode = "MODE_X_Y_Z"

                if mode == "MODE_X":
                    distance_top = dx_top
                    distance_low = dx_low
                elif mode == "MODE_X_Y":
                    distance_top = math.sqrt(dx_top**2 + dy_top**2)
                    distance_low = math.sqrt(dx_low**2 + dy_low**2)
                elif mode == "MODE_Y":
                    distance_top = dy_top
                    distance_low = dy_low
                elif mode == "MODE_X_Y_Z":
                    distance_top = math.sqrt(dx_top**2 + dy_top**2 + dz_top**2)
                    distance_low = math.sqrt(dx_low**2 + dy_low**2 + dz_low**2)
                else:
                    distance_top = math.sqrt(dx_top**2 + dy_top**2 + dz_top**2)
                    distance_low = math.sqrt(dx_low**2 + dy_low**2 + dz_low**2)

                # === Deteksi arah & peak (kept identical) ===
                sign = [None] * 2
                distances = [distance_top, distance_low]
                for i in range(2):
                    try:
                        sign[i] = 1 if distances[i] > 0 else -1
                    except Exception:
                        sign[i] = 1 if (distances[i] and distances[i] > 0) else -1

                    try:
                        self.temp_values[i].append(distances[i])
                    except Exception:
                        pass

                    if (sign[i] != self.last_sign[i] and len(self.temp_values[i]) > 2 or len(self.temp_values[i]) >= (MAX_TEMP_VALUES-1)):
                        if self.last_sign[i] != 0:
                            if self.last_sign[i] > 0:
                                self.peak_value[i] = max(list(self.temp_values[i])[:-2])
                            elif self.last_sign[i] < 0:
                                self.peak_value[i] = min(list(self.temp_values[i])[:-2])

                            marker_name = "top marker" if i == 0 else "bottom marker"
                            print(f"Peak = {self.peak_value[i]:.2f} mm ({marker_name})")

                        self.last_sign[i] = sign[i]
                        self.temp_values[i].clear()

                        if i == 1:
                            data_entry = {
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "laptop_id": self.laptop_id,
                                "is_a_detected": bool(self.tracker.markers.get(base + 5) and self.tracker.markers[base + 5].detected),
                                "is_b_detected": bool(self.tracker.markers.get(base + 4) and self.tracker.markers[base + 4].detected),
                                "displacement_a": float(self.peak_value[0]) if self.peak_value[0] is not None else None,
                                "displacement_b": float(self.peak_value[1]) if self.peak_value[1] is not None else None
                            }
                            # synchronous write as original
                            with save_lock:
                                try:
                                    with open("displacement.json", "w") as jf:
                                        json.dump(data_entry, jf, indent=4)
                                except Exception as e:
                                    print(f"Write displacement.json failed: {e}")
                                try:
                                    file_exists = os.path.isfile("displacement_log.csv")
                                    with open("displacement_log.csv", "a", newline="") as cf:
                                        writer = csv.DictWriter(cf, fieldnames=data_entry.keys())
                                        if not file_exists:
                                            writer.writeheader()
                                        writer.writerow(data_entry)
                                except Exception as e:
                                    print(f"Append displacement_log.csv failed: {e}")

            if self.prev_timer[0] != 0.0 and (time.time() - self.prev_timer[0]) > 1.0:
                self.last_sign[0] = 0
                self.peak_value[0] = None
                self.temp_values[0].clear()
            if self.prev_timer[1] != 0.0 and (time.time() - self.prev_timer[1]) > 1.0:
                self.last_sign[1] = 0
                self.peak_value[1] = None
                self.temp_values[1].clear()

            # overlay peaks
            if self.peak_value[0] is not None:
                try:
                    cv2.putText(frame, f"peak displacement top ={self.peak_value[0]:.2f}mm", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                except Exception:
                    pass
            if self.peak_value[1] is not None:
                try:
                    cv2.putText(frame, f"peak displacement bottom ={self.peak_value[1]:.2f}mm", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                except Exception:
                    pass

            # add line intersection x = 0 y = 0
            cv2.line(frame, (FRAME_WIDTH // 2, 0), (FRAME_WIDTH // 2, FRAME_HEIGHT), (255, 255, 0), 1)
            cv2.line(frame, (0, FRAME_HEIGHT // 2), (FRAME_WIDTH, FRAME_HEIGHT // 2), (255, 255, 0), 1)
            # show frame
            try:
                cv2.imshow("Marker Detection", frame)
            except Exception:
                pass

            # read key (non-blocking); if ESC pressed, request stop
            k = cv2.waitKey(1)
            if k == 27:
                self._stop_event.set()
                break

        # cleanup on exit
        try:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
        except Exception:
            pass

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    def calibrate_current(self):
        # calibration uses BASE_MARKER mapping purely
        base = (self.laptop_id - 1) * 5
        for i in range(1, 6):
            real_index = base + i
            if real_index in self.tracker.markers and self.tracker.markers[real_index].detected:
                src = self.tracker.markers[real_index]
                self.calibrate_values_x[i] = src.x
                self.calibrate_values_y[i] = src.y
                self.calibrate_values_z[i] = src.z
                self.calibrated[i] = True
            else:
                self.calibrated[i] = False

class DataSender(threading.Thread):
    def __init__(self, camera_handler, 
                 api_url="http://localhost/detector-getaran/api/receive_camera_data.php",
                 retry_interval=1.0,
                 max_buffer=100):
        """
        :param camera_handler: referensi ke CameraHandler (punya displacement dan deteksi)
        :param api_url: alamat PHP endpoint
        :param retry_interval: waktu tunggu sebelum retry jika gagal
        :param max_buffer: jumlah maksimal data yang disimpan lokal
        """
        super().__init__(daemon=True)
        self.camera_handler = camera_handler
        self.api_url = api_url
        self.retry_interval = retry_interval
        self.buffer = deque(maxlen=max_buffer)
        self._stop_event = threading.Event()
        self.last_sent_data = None

    def stop(self):
        self._stop_event.set()

    def _build_data(self):
        """Ambil data terbaru dari kamera"""
        return {
            "laptop_id": self.camera_handler.laptop_id,
            "dista": float(self.camera_handler.peak_value[0]) if self.camera_handler.peak_value[0] is not None else 0.0,
            "distb": float(self.camera_handler.peak_value[1]) if self.camera_handler.peak_value[1] is not None else 0.0,
            "is_a_detected": bool(
                self.camera_handler.tracker.markers.get((self.camera_handler.laptop_id - 1) * 5 + 5)
                and self.camera_handler.tracker.markers[(self.camera_handler.laptop_id - 1) * 5 + 5].detected
            ),
            "is_b_detected": bool(
                self.camera_handler.tracker.markers.get((self.camera_handler.laptop_id - 1) * 5 + 4)
                and self.camera_handler.tracker.markers[(self.camera_handler.laptop_id - 1) * 5 + 4].detected
            )
        }

    def _send_to_server(self, data):
        """Kirim ke server PHP, return True kalau sukses"""
        try:
            response = requests.post(self.api_url, json=data, timeout=1)
            if response.status_code == 200:
                print(f"✅ Sent: {data}")
                return True
            else:
                print(f"⚠️ Server returned {response.status_code}: {response.text[:100]}")
                return False
        except requests.RequestException as e:
            print(f"❌ Send error: {e}")
            return False

    def run(self):
        while not self._stop_event.is_set():
            # ambil data terbaru dari kamera
            new_data = self._build_data()

            # hanya tambahkan ke buffer jika berbeda dari sebelumnya
            if new_data != self.last_sent_data:
                self.buffer.append(new_data)
                self.last_sent_data = new_data.copy()

            # jika buffer ada isinya, kirim dari yang paling lama
            if self.buffer:
                current = self.buffer[0]
                if self._send_to_server(current):
                    self.buffer.popleft()  # sukses, hapus dari buffer
                else:
                    # gagal kirim → tunggu sebentar sebelum mencoba lagi
                    time.sleep(self.retry_interval)
                    continue

            time.sleep(0.01)  # mencegah CPU 100%

# ====== GUI (Tkinter) - starts/stops the CameraHandler thread ======
class MarkerGUI:
    def __init__(self, root):
        self.root = root
        root.title("Marker Control Panel - BaseID Refactor")
        root.geometry("380x460")

        self.camera_index = CAMERA_INDEX_DEFAULT
        self.camera_thread = None

        tk.Label(root, text="Laptop ID:").pack(pady=5)
        self.entry_id = tk.Entry(root, width=10)
        self.entry_id.insert(0, "1")
        self.entry_id.pack(pady=5)

        tk.Label(root, text="Camera Index:").pack(pady=5)
        self.entry_cam = tk.Entry(root, width=10)
        self.entry_cam.insert(0, str(CAMERA_INDEX_DEFAULT))
        self.entry_cam.pack(pady=5)

        tk.Label(root, text="Select Mode:").pack(pady=5)
        self.mode_var = tk.StringVar(value="MODE_X_Y_Z")
        modes = [("MODE_X", "MODE_X"), ("MODE_X_Y", "MODE_X_Y"), ("MODE_Y", "MODE_Y"), ("MODE_X_Y_Z", "MODE_X_Y_Z")]
        for text, mode in modes:
            tk.Radiobutton(root, text=text, variable=self.mode_var, value=mode).pack(anchor=tk.W)

        self.btn_start = tk.Button(root, text="Start Detection", bg="lightblue", command=self.start_detection)
        self.btn_start.pack(pady=6)

        self.btn_cal = tk.Button(root, text="Calibrate", bg="lightgreen", command=self.do_calibrate)
        self.btn_cal.pack(pady=6)

        self.btn_exit = tk.Button(root, text="Exit", bg="tomato", command=self.exit_program)
        self.btn_exit.pack(pady=6)

        self.label_status = tk.Label(root, text="Status: Idle", fg="blue")
        self.label_status.pack(pady=10)

        # UI update
        self.update_ui()

    def start_detection(self):
        try:
            laptop_id = int(self.entry_id.get())
        except ValueError:
            messagebox.showwarning("Input Error", "Laptop ID harus berupa angka!")
            return
        try:
            cam_idx = int(self.entry_cam.get())
        except ValueError:
            messagebox.showwarning("Input Error", "Camera index harus angka!")
            return

        # create and start camera thread
        self.camera_thread = CameraHandler(camera_index=cam_idx, laptop_id=laptop_id)
        self.camera_thread.mode_var = self.mode_var
        self.camera_thread.start()

        self.DataSenderThread = DataSender(camera_handler=self.camera_thread)
        self.DataSenderThread.start()

        self.label_status.config(text=f"Status: Detection Running (ID={laptop_id})", fg="green")

    def do_calibrate(self):
        if not self.camera_thread or not self.camera_thread.is_alive():
            messagebox.showwarning("Not Ready", "Detection belum dimulai!")
            return
        time.sleep(0.3)
        self.camera_thread.calibrate_current()
        self.label_status.config(text="Status: Calibrated", fg="orange")
        messagebox.showinfo("Calibration", "Kalibrasi selesai!")

    def exit_program(self):
        # safe shutdown: signal thread to stop and join briefly
        if hasattr(self, "DataSenderThread") and self.DataSenderThread.is_alive():
            self.DataSenderThread.stop()
            self.DataSenderThread.join(timeout=2.0)


        if self.camera_thread and self.camera_thread.is_alive():
            try:
                self.camera_thread.stop()
            except Exception:
                pass
            self.camera_thread.join(timeout=2.0)

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        try:
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass

    def update_ui(self):
        if self.camera_thread:
            snap = self.camera_thread.tracker.markers
            if snap:
                s = "Markers: " + ", ".join([f"{mid}:{'1' if m.detected else '0'}" for mid, m in snap.items()])
                self.label_status.config(text=s, fg="green")
        self.root.after(500, self.update_ui)

if __name__ == "__main__":
    root = tk.Tk()
    app = MarkerGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.exit_program)
    root.mainloop()
