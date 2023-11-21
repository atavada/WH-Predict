# Deskripsi: Program untuk memprediksi berat badan berdasarkan jarak antara dua bahu dan tinggi badan dengan perhitungan integral tabung

# Import library yang dibutuhkan
import cv2
import numpy as np
import mediapipe as mp
from scipy.integrate import quad

# Load model Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Open webcam
cap = cv2.VideoCapture(0)

# Jarak kita terhadap kamera (dalam cm)
camera_distance = 200

# Inisialisasi variabel untuk menyimpan posisi y tulisan
text_y_positions = {
    "Shoulder Distance": 40,
    "Shoulder Radius": 60,
    "Estimated Height": 100,
    "Estimated Weight": 80,
}

# Gambar titik
def draw_point(image, point, color):
    cv2.circle(image, (int(point[0]), int(point[1])), 5, color, cv2.FILLED)

# Gambar garis
def draw_line(image, point1, point2, color):
    cv2.line(image, (int(point1[0]), int(point1[1])), (int(point2[0]), int(point2[1])), color, 2)

# Hitung jarak antara dua titik
def calculate_distance(point1, point2):
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

# Hitung volume dan berat badan
def calculate_volume_and_weight(radius_cm, height_cm):
    r_squared = radius_cm**2  # Convert cm to meters
    def integrand(y):
        return np.pi * r_squared
    volume, _ = quad(integrand, 0, height_cm)  # Convert cm to meters
    estimated_weight_kg = 2/3 * volume / 1000  # Convert grams to kilograms
    return volume, estimated_weight_kg

# Looping untuk membaca frame dari webcam
while True:
    # Baca frame dari webcam
    ret, frame = cap.read()

    # Deteksi pose menggunakan Mediapipe Pose
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    # Gambar titik-titik pose pada frame
    if results.pose_landmarks:
        shoulder_left = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * frame.shape[1], results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame.shape[0])
        shoulder_right = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * frame.shape[1], results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame.shape[0])
        ankle_right = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x * frame.shape[1], results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y * frame.shape[0])

        # Gambar titik dan garis penghubung
        draw_point(frame, shoulder_left, (255, 0, 0))
        draw_point(frame, shoulder_right, (255, 0, 0))
        draw_point(frame, ankle_right, (255, 0, 0))
        draw_line(frame, shoulder_left, shoulder_right, (0, 255, 0))
        draw_line(frame, shoulder_right, ankle_right, (0, 255, 0))

        # Pengukuran lebar bahu dan tinggi tubuh
        shoulder_distance_cm = calculate_distance(shoulder_left, shoulder_right) / frame.shape[1] * camera_distance
        height_measure_cm = np.abs(ankle_right[1] - shoulder_right[1]) / frame.shape[0] * camera_distance
        real_height = height_measure_cm + 30

        # Perhitungan volume dan berat badan
        shoulder_radius_cm = shoulder_distance_cm / 2
        volume, estimated_weight_kg = calculate_volume_and_weight(shoulder_radius_cm, real_height)

        # Tampilkan hasil pengukuran
        for measurement, value in zip(["Shoulder Distance", "Shoulder Radius", "Estimated Height", "Estimated Weight"], [shoulder_distance_cm, shoulder_radius_cm, real_height, estimated_weight_kg]):
            cv2.putText(
                frame,
                f"{measurement}: {value:.2f} {'cm' if measurement in ['Shoulder Distance', 'Shoulder Radius', 'Estimated Height'] else 'kg'}",
                (10, text_y_positions[measurement]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
            )

    # Tampilkan frame
    cv2.imshow("Webcam", frame)

    # Hentikan program dengan menekan tombol 'ESC'
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord(" "):  # Tombol spasi ditekan
        # Buat jendela baru untuk menampilkan berat badan dan tinggi badan
        result_window = np.ones((200, 400, 3), dtype=np.uint8) * 255  # Jendela berukuran 400x200 pixel
        result_text_y = 20

        # Tampilkan hasil pengukuran pada jendela baru
        for measurement, value in zip(["Shoulder Distance", "Shoulder Radius", "Estimated Height", "Estimated Weight"], [shoulder_distance_cm, shoulder_radius_cm, real_height, estimated_weight_kg]):
            cv2.putText(
                result_window,
                f"{measurement}: {value:.2f} {'cm' if measurement in ['Shoulder Distance', 'Shoulder Radius', 'Estimated Height'] else 'kg'}",
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
