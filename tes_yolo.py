import cv2
from ultralytics import YOLO
import time

# 1. Load Model
# Saat pertama kali dijalankan, dia akan otomatis download file 'yolov8n-pose.pt' (sekitar 6MB)
print("Sedang memuat model YOLOv8-pose...")
model = YOLO('yolov8n-pose.pt') 

# 2. Buka Video
video_path = 'tes_video.mp4' # Pastikan nama file sesuai!
cap = cv2.VideoCapture(video_path)

# Cek apakah video berhasil dibuka
if not cap.isOpened():
    print(f"Error: Tidak bisa membuka video '{video_path}'. Cek nama file/lokasi!")
    exit()

print("Mulai mendeteksi... Tekan 'q' di keyboard untuk berhenti.")

while True:
    success, frame = cap.read()
    
    if not success:
        print("Video selesai.")
        break

    # 3. Jalankan YOLO pada frame ini
    # conf=0.5 artinya hanya tampilkan jika yakin > 50%
    results = model(frame, conf=0.5)

    # 4. Visualisasi (Menggambar skeleton otomatis)
    annotated_frame = results[0].plot()

    # 5. Tampilkan di Layar Pop-up
    cv2.imshow('Tes YOLOv8 Pose - Tekan q untuk keluar', annotated_frame)

    # Logika keluar (tekan q)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()