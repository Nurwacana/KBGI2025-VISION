import cv2
import cv2.aruco as aruco
import numpy as np
import time
import json


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
        """Reset semua nilai ke nol"""
        self.x = self.y = self.z = 0.0
        self.roll = self.pitch = self.yaw = 0.0

    def __repr__(self):
        return (f"Marker(id={self.id}, detected={self.detected}, "
                f"pos=({self.x:.1f},{self.y:.1f},{self.z:.1f}), "
                f"rot=({self.roll:.1f},{self.pitch:.1f},{self.yaw:.1f}))")
    
class offsetMarker:
    def __init__(self, id_):
        self.id = id_
        self.detected = False
        self.distance = 0.0


# ==== Tracker ====
class MarkerTracker:
    def __init__(self, timeout=1.0):
        self.markers = {}
        self.timeout = timeout

    def __getitem__(self, id_):
        if id_ not in self.markers:
            self.markers[id_] = Marker(id_)
        return self.markers[id_]

    def update_marker(self, id_, corners=None, rvec=None, tvec=None):
        """Update marker saat terdeteksi"""
        m = self[id_]
        R, _ = cv2.Rodrigues(rvec)
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        m.roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        m.pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        m.yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        m.x, m.y, m.z = tvec[0] * 1000
        m.detected = True
        m.last_seen = time.time()

        # posisi tengah marker
        c = corners[0]
        center = np.mean(c, axis=0).astype(int)
        m.last_pos_2d = tuple(center)

    def mark_all_undetected(self):
        """Tandai semua marker sementara sebagai tidak terdeteksi"""
        for m in self.markers.values():
            m.detected = False

    def apply_timeout(self):
        """Reset marker jika tidak terlihat dalam jangka waktu tertentu"""
        now = time.time()
        for m in self.markers.values():
            if not m.detected and (now - m.last_seen > self.timeout):
                m.reset()

def marker_to_dict(m):
    return {
        "id": int(m.id),
        "detected": bool(m.detected),
        "x": float(m.x),
        "y": float(m.y),
        "z": float(m.z),
        "roll": float(m.roll),
        "pitch": float(m.pitch),
        "yaw": float(m.yaw),
        "last_seen": float(m.last_seen),
        "last_pos_2d": [int(m.last_pos_2d[0]), int(m.last_pos_2d[1])]
    }

def offset_to_dict(o):
    return {
        "id": int(o.id),
        "detected": bool(o.detected),
        "distance": float(o.distance)
    }


# ==== Setup kamera ====
cameraMatrix = np.array([[640, 0, 640],
                         [0, 640, 360],
                         [0, 0, 1]], dtype=np.float32)
distCoeffs = np.zeros((5, 1))
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

while True:
    ret, frame = cap.read()
    if not ret:
        break

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

    # reset marker yang sudah lewat timeout
    tracker.apply_timeout()

    # tampilkan hanya marker yang terdeteksi pada frame saat ini
    for m in tracker.markers.values():
        if not m.detected:
            continue  # skip marker yang tidak terlihat di frame ini

        cx, cy = m.last_pos_2d
        color = (0, 255, 0)
        text = (f"ID:{m.id} "
                f"X:{m.x:.0f} Y:{m.y:.0f} Z:{m.z:.0f} "
                f"R:{m.roll:.0f} P:{m.pitch:.0f} Y:{m.yaw:.0f}")
        cv2.putText(frame, text, (cx - 60, cy - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # calculate distance x, y between marker 1 and 2 if both detected
    if tracker[1].detected and tracker[2].detected:
        dx = tracker[2].x - tracker[1].x
        dy = tracker[2].y - tracker[1].y
        # dz = tracker[2].z - tracker[1].z
        # dist = np.sqrt(dx**2 + dy**2 + dz**2)
        dist = np.sqrt(dx**2 + dy**2)
        offset_1_2.distance = dist
        print(f"Distance between Marker 1 and 2: {dist:.1f} mm")
        cv2.line(frame, tracker[1].last_pos_2d, tracker[2].last_pos_2d, (255, 0, 0), 2)

    # calculate distance x, y between marker 1 and 3 if both detected
    if tracker[1].detected and tracker[3].detected:
        dx = tracker[3].x - tracker[1].x
        dy = tracker[3].y - tracker[1].y
        # dz = tracker[2].z - tracker[1].z
        # dist = np.sqrt(dx**2 + dy**2 + dz**2)
        dist = np.sqrt(dx**2 + dy**2)
        offset_1_3.distance = dist
        print(f"Distance between Marker 1 and 2: {dist:.1f} mm")

        cv2.line(frame, tracker[1].last_pos_2d, tracker[3].last_pos_2d, (0, 255, 0), 2)

        # state = {str(id): offset_to_dict(offset_1_2) for id in [offset_1_2.id]}
        # state.update({str(id): offset_to_dict(offset_1_3) for id in [offset_1_3.id]})

        # with open("markers.json", "w") as f:
        #     json.dump(state, f, indent=4)

    cv2.imshow('Marker Detection', frame)
    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
