import cv2
import math
import numpy as np
from ultralytics import YOLO

# --- KONFIGURASI ---
VIDEO_SOURCE = 'tes_video.mp4' # Ganti dengan nama videomu
MODEL_PATH = 'yolov8n-pose.pt'
CONFIDENCE_THRESHOLD = 0.5 
FALL_ANGLE_THRESHOLD = 45 # Derajat (di bawah ini dianggap jatuh)

# --- FUNGSI MENGHITUNG SUDUT ---
def calculate_angle(p1, p2):
    # p1 = [x, y] bahu, p2 = [x, y] pinggul
    x1, y1 = p1
    x2, y2 = p2
    
    # Hitung kemiringan (slope)
    # Jika x sama, berarti tegak lurus (90 derajat)
    if x2 - x1 == 0:
        return 90.0
    
    # Rumus Trigonometri Dasar
    slope = (y2 - y1) / (x2 - x1)
    angle_rad = math.atan(slope)
    angle_deg = math.degrees(angle_rad)
    
    return abs(angle_deg)

# --- LOAD MODEL ---
print("Memuat Model...")
model = YOLO(MODEL_PATH)

# --- BUKA VIDEO ---
cap = cv2.VideoCapture(VIDEO_SOURCE)

while True:
    success, frame = cap.read()
    if not success:
        print("Video Selesai.")
        break
    
    # Jalankan YOLO
    results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
    
    for result in results:
        # Ambil data Keypoints (Sendi-sendi)
        # Bentuk data: [jumlah_orang, 17 titik, 3 data (x, y, conf)]
        keypoints = result.keypoints.data.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy() # Bounding Box
        
        for i, person_kps in enumerate(keypoints):
            # Ambil koordinat penting (Sesuai format COCO)
            # 5=Bahu Kiri, 6=Bahu Kanan, 11=Pinggul Kiri, 12=Pinggul Kanan
            l_shoulder = person_kps[5][:2]
            r_shoulder = person_kps[6][:2]
            l_hip = person_kps[11][:2]
            r_hip = person_kps[12][:2]
            
            # Cek kepercayaan (confidence) deteksi sendi
            # Jika sendi tidak terdeteksi jelas, jangan dihitung (biar tidak error)
            if person_kps[5][2] > 0.5 and person_kps[6][2] > 0.5 and \
               person_kps[11][2] > 0.5 and person_kps[12][2] > 0.5:
                
                # 1. Cari Titik Tengah Bahu & Pinggul
                shoulder_mid_x = int((l_shoulder[0] + r_shoulder[0]) / 2)
                shoulder_mid_y = int((l_shoulder[1] + r_shoulder[1]) / 2)
                
                hip_mid_x = int((l_hip[0] + r_hip[0]) / 2)
                hip_mid_y = int((l_hip[1] + r_hip[1]) / 2)
                
                # 2. Hitung Sudut Tulang Belakang
                angle = calculate_angle((shoulder_mid_x, shoulder_mid_y), (hip_mid_x, hip_mid_y))
                
                # 3. Analisis Bounding Box (Kotak)
                # Box format: x1, y1, x2, y2
                box = boxes[i]
                width = box[2] - box[0]
                height = box[3] - box[1]
                
                # --- LOGIKA PENENTUAN STATUS ---
                # Default status
                status = "Normal"
                color = (0, 255, 0) # Hijau
                
                # JIKA Sudut < 45 derajat ATAU Lebar > Tinggi
                if angle < FALL_ANGLE_THRESHOLD or width > height:
                    status = "JATUH TERDETEKSI!"
                    color = (0, 0, 255) # Merah (Format BGR: Blue, Green, Red)
                
                # --- VISUALISASI ---
                
                # Gambar Garis Tulang Belakang
                cv2.line(frame, (shoulder_mid_x, shoulder_mid_y), (hip_mid_x, hip_mid_y), (255, 255, 0), 2)
                
                # Gambar Kotak di sekeliling orang
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Tulis Status & Sudut di atas kepala
                info_text = f"{status} (Sudut: {int(angle)} deg)"
                cv2.putText(frame, info_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Tampilkan
    cv2.imshow("Sistem Deteksi Jatuh - V1", frame)
    
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()