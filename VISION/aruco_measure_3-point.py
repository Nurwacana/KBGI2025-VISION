import cv2
import cv2.aruco as aruco
import numpy as np

# ==== Dummy Camera Calibration (sementara) ====
cameraMatrix = np.array([[640, 0, 640],
                         [0, 640, 360],
                         [0, 0, 1]], dtype=np.float32)
distCoeffs = np.zeros((5, 1))

# ==== Kamera Setup ====
cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
# cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # pakai DirectShow di Windows
width = 1280
height = 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FPS, 60)

# cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 = manual mode (Windows / UVC)
# cap.set(cv2.CAP_PROP_EXPOSURE, -9)         # contoh nilai manual (tergantung kamera)
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

print("Camera properties:")
print("Width:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("Height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Exposure:", cap.get(cv2.CAP_PROP_EXPOSURE))
print("Auto exposure:", cap.get(cv2.CAP_PROP_AUTO_EXPOSURE))
print("FPS:", cap.get(cv2.CAP_PROP_FPS))

# ==== ArUco Setup (API baru OpenCV 4.10+) ====
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

# ==== Loop utama ====
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # brighten frame jika perlu
    # frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # deteksi marker (pakai ArucoDetector)
    corners, ids, rejected = detector.detectMarkers(gray)


    if ids is not None:
        # gambar outline marker
        aruco.drawDetectedMarkers(frame, corners, ids)

        # estimasi pose marker (ukuran marker 8 cm)
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.08, cameraMatrix, distCoeffs)

        for i in range(len(ids)):
            # # patokan meja getar gedung
            # if i == 0:


            # Dapatkan rotasi dan translasi
            R, _ = cv2.Rodrigues(rvec[i])
            sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
            roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
            pitch = np.degrees(np.arctan2(-R[2, 0], sy))
            yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))

            # Ambil posisi (X, Y, Z) dalam meter → ubah ke cm
            x, y, z = tvec[i][0] * 1000

            # Gambar axis 3D (pakai fungsi baru cv2.drawFrameAxes)
            cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvec[i], tvec[i], 0.03)

            # Tulis posisi dan orientasi di frame
            corner = corners[i][0][0].astype(int)
            text = f"ID:{ids[i][0]} X:{x:.1f} Y:{y:.1f} Z:{z:.1f} cm"
            cv2.putText(frame, text, (corner[0], corner[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Cetak di terminal
            print(f"Marker {ids[i][0]} → Roll={roll:.2f}°, Pitch={pitch:.2f}°, Yaw={yaw:.2f}°, X={x:.2f} mm, Y={y:.2f} mm, Z={z:.2f} mm, time={cv2.getTickCount()/cv2.getTickFrequency():.2f}s")

    cv2.imshow('Marker Detection', frame)
    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
