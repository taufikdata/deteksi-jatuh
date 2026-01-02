import cv2
import math
import os
import datetime
from ultralytics import YOLO

# --- KONFIGURASI ---
VIDEO_SOURCE = 'tes_video.mp4'
MODEL_PATH = 'yolov8n-pose.pt'
CONFIDENCE_THRESHOLD = 0.5
FALL_ANGLE_THRESHOLD = 45

# Konfigurasi Baru: Counter
FALL_TIME_THRESHOLD = 5  # Harus terdeteksi jatuh selama 20 frame berturut-turut baru alarm bunyi
fall_counter = 0          # Penghitung frame jatuh
alarm_active = False      # Status alarm

# Konfigurasi Baru: Folder Penyimpanan Bukti
save_dir = "bukti_jatuh"
if not os.path.exists(save_dir):
    os.makedirs(save_dir) # Buat folder otomatis jika belum ada

# --- FUNGSI HITUNG SUDUT (Sama seperti V1) ---
def calculate_angle(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    if x2 - x1 == 0: return 90.0
    slope = (y2 - y1) / (x2 - x1)
    angle_deg = math.degrees(math.atan(slope))
    return abs(angle_deg)

# --- SETUP ---
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_SOURCE)

# Ambil FPS untuk delay
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)

print("Sistem Deteksi Jatuh V2 Berjalan...")

while True:
    success, frame = cap.read()
    if not success: break
    
    # Copy frame untuk digambar (visualisasi), frame asli tetap bersih untuk screenshot
    vis_frame = frame.copy()
    
    results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
    
    # Default: Asumsi tidak ada yang jatuh di frame ini
    is_currently_falling = False 

    for result in results:
        keypoints = result.keypoints.data.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()
        
        for i, person_kps in enumerate(keypoints):
            # Cek deteksi bahu & pinggul valid
            if person_kps[5][2] > 0.5 and person_kps[6][2] > 0.5 and \
               person_kps[11][2] > 0.5 and person_kps[12][2] > 0.5:
                
                # Koordinat Bahu & Pinggul
                shoulder_mid_x = int((person_kps[5][0] + person_kps[6][0]) / 2)
                shoulder_mid_y = int((person_kps[5][1] + person_kps[6][1]) / 2)
                hip_mid_x = int((person_kps[11][0] + person_kps[12][0]) / 2)
                hip_mid_y = int((person_kps[11][1] + person_kps[12][1]) / 2)
                
                angle = calculate_angle((shoulder_mid_x, shoulder_mid_y), (hip_mid_x, hip_mid_y))
                
                # Cek Box Ratio
                box = boxes[i]
                width = box[2] - box[0]
                height = box[3] - box[1]
                
                # Visualisasi Skeleton (Garis Kuning)
                cv2.line(vis_frame, (shoulder_mid_x, shoulder_mid_y), (hip_mid_x, hip_mid_y), (0, 255, 255), 2)
                
                # --- LOGIKA DETEKSI ---
                if angle < FALL_ANGLE_THRESHOLD or width > height:
                    is_currently_falling = True
                    # Gambar kotak Merah
                    cv2.rectangle(vis_frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
                else:
                    # Gambar kotak Hijau
                    cv2.rectangle(vis_frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

    # --- LOGIKA COUNTER & ALARM (PENTING) ---
    if is_currently_falling:
        fall_counter += 1
    else:
        fall_counter = 0 # Reset jika orangnya berdiri lagi
        alarm_active = False

    # Jika jatuh terdeteksi terus-menerus melebihi batas waktu
    if fall_counter > FALL_TIME_THRESHOLD:
        alarm_active = True
        
        # Tampilkan Peringatan Besar
        cv2.putText(vis_frame, "PERINGATAN: JATUH TERDETEKSI!!!", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        # FITUR: Simpan Bukti Foto (Hanya sekali saat awal trigger alarm)
        # Kita pakai modulo biar gak nyimpen ribuan foto, misal simpan tiap 30 frame sekali selama jatuh
        if fall_counter == FALL_TIME_THRESHOLD + 1:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{save_dir}/Jatuh_{timestamp}.jpg"
            cv2.imwrite(filename, frame) # Simpan frame asli (bersih)
            print(f"Bukti tersimpan: {filename}")

    # Tampilkan Counter di layar (untuk debug/monitoring)
    cv2.putText(vis_frame, f"Fall Counter: {fall_counter}/{FALL_TIME_THRESHOLD}", (10, vis_frame.shape[0]-20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Sistem Peringatan Dini - V2", vis_frame)
    
    key = cv2.waitKey(100) & 0xFF
    if key == ord('q'): break

cap.release()
cv2.destroyAllWindows()