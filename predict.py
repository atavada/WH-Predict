import cv2
import numpy as np
import mediapipe as mp
from scipy.integrate import quad
from math import pi

# Load model Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Open webcam
cap = cv2.VideoCapture(0)

# Jarak kita terhadap kamera (dalam cm)
jarak_kamera = 200

# Inisialisasi variabel untuk menyimpan posisi y tulisan
text_y_positions = {
    "Shoulder Distance": 40,
    "Shoulder Radius": 60,
    "Tinggi Badan": 100,
    "Berat Badan": 80,
}

while True:
    # Baca frame dari webcam
    ret, frame = cap.read()

    # Deteksi pose menggunakan Mediapipe Pose
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    # Gambar titik-titik pose pada frame
    if results.pose_landmarks:
        # Gambar titik bahu kiri
        shoulder_left = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * frame.shape[1], results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame.shape[0])
        cv2.circle(frame, (int(shoulder_left[0]), int(shoulder_left[1])), 5, (255, 0, 0), cv2.FILLED)

        # Gambar titik bahu kanan
        shoulder_right = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * frame.shape[1], results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame.shape[0])
        cv2.circle(frame, (int(shoulder_right[0]), int(shoulder_right[1])), 5, (255, 0, 0), cv2.FILLED)

        # Gambar titik ujung kaki kanan
        ankle_right = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x * frame.shape[1], results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y * frame.shape[0])
        cv2.circle(frame, (int(ankle_right[0]), int(ankle_right[1])), 5, (255, 0, 0), cv2.FILLED)

        # Pengukuran lebar bahu kiri ke kanan (setengah dari ukuran aslinya)
        shoulder_distance_cm = np.sqrt((shoulder_right[0] - shoulder_left[0])**2 + (shoulder_right[1] - shoulder_left[1])**2) / frame.shape[1] * (jarak_kamera - 30)

        # Pengukuran tinggi dari bahu kanan ke ujung kaki kanan
        height_measure_cm = np.abs(ankle_right[1] - shoulder_right[1]) / frame.shape[0] * (jarak_kamera - 30)
        height_measure_cm1 = np.abs(ankle_right[1] - shoulder_right[1]) / frame.shape[0] * (jarak_kamera - 30) + 40

        # Radius bahu (setengah dari panjang asli lebar bahu)
        shoulder_radius_cm = shoulder_distance_cm / 2

        # Hitung volume tabung
        def integrand(y):
            r_squared = (shoulder_radius_cm) ** 2  # Convert cm to meters
            return pi * r_squared

        volume, _ = quad(integrand, 0, height_measure_cm)  # Convert cm to meters

        # Estimasi berat badan (density manusia ~1 kg/L)
        estimated_weight_kg = volume   # Convert to grams
        estimated_weight_kg /= 1000  # Convert grams to kilograms

        # Tampilkan hasil pengukuran
        for measurement, value in zip(["Shoulder Distance", "Shoulder Radius", "Tinggi Badan", "Berat Badan"], [shoulder_distance_cm, shoulder_radius_cm, height_measure_cm1, estimated_weight_kg]):
            cv2.putText(
                frame,
                f"{measurement}: {value:.2f} {'cm' if measurement in ['Shoulder Distance', 'Shoulder Radius', 'Tinggi Badan'] else 'kg'}",
                (10, text_y_positions[measurement]),  # Updated coordinates for top-left corner
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
            )

    # Tampilkan frame
    cv2.imshow("Webcam", frame)

    # Hentikan program dengan menekan tombol 'ESC'
    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key == ord(" "):  # Tombol spasi ditekan
        # Buat jendela baru untuk menampilkan berat badan dan tinggi badan
        result_window = np.ones((200, 400, 3), dtype=np.uint8) * 255  # Jendela berukuran 400x200 pixel
        result_text_y = 20

        # Tampilkan hasil pengukuran pada jendela baru
        for measurement, value in zip(["Shoulder Distance", "Shoulder Radius", "Tinggi Badan", "Berat Badan"], [shoulder_distance_cm, shoulder_radius_cm, height_measure_cm1, estimated_weight_kg]):
            cv2.putText(
                result_window,
                f"{measurement}: {value:.2f} {'cm' if measurement in ['Shoulder Distance', 'Shoulder Radius', 'Tinggi Badan'] else 'kg'}",
                (10, result_text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
            )
            result_text_y += 40

        # Tampilkan jendela baru
        cv2.imshow("Results", result_window)

# Tutup webcam dan jendela OpenCV
cap.release()
cv2.destroyAllWindows()