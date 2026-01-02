import cv2
import os
import pandas as pd
import math
from ultralytics import YOLO

# --- KONFIGURASI PATH ---
PATH_FALLS = 'dataset/falls' 
PATH_ADL = 'dataset/adl'     
OUTPUT_IMAGE_DIR = 'hasil_visual_uji' # Folder baru untuk simpan bukti uji

# Buat folder output jika belum ada
if not os.path.exists(OUTPUT_IMAGE_DIR):
    os.makedirs(OUTPUT_IMAGE_DIR)

# Konfigurasi Model
MODEL_PATH = 'yolov8n-pose.pt'
CONF_THRESHOLD = 0.5
FALL_TIME_THRESHOLD = 5  
FALL_ANGLE = 45

# --- FUNGSI DETEKSI (Updated) ---
def detect_fall_in_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    fall_counter = 0
    is_fall_detected = False
    
    # Ambil nama file untuk penamaan foto bukti
    filename = os.path.basename(video_path)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        results = model(frame, conf=CONF_THRESHOLD, verbose=False)
        frame_fall_status = False
        
        # Simpan frame original untuk bukti (bersih tanpa coretan)
        evidence_frame = frame.copy()
        
        for result in results:
            if result.keypoints is None or len(result.keypoints) == 0:
                continue

            kps = result.keypoints.data.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()
            
            for i, person in enumerate(kps):
                if person[5][2] < 0.5 or person[6][2] < 0.5 or \
                   person[11][2] < 0.5 or person[12][2] < 0.5:
                    continue
                
                shoulder_y = (person[5][1] + person[6][1]) / 2
                shoulder_x = (person[5][0] + person[6][0]) / 2
                hip_y = (person[11][1] + person[12][1]) / 2
                hip_x = (person[11][0] + person[12][0]) / 2
                
                dx = hip_x - shoulder_x
                dy = hip_y - shoulder_y
                
                if dx == 0: angle = 90
                else: angle = abs(math.degrees(math.atan(dy/dx)))
                
                box = boxes[i]
                w = box[2] - box[0]
                h = box[3] - box[1]
                
                if angle < FALL_ANGLE or w > h:
                    frame_fall_status = True
                    
                    # (Opsional) Gambar kotak merah di frame bukti biar jelas
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(evidence_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(evidence_frame, f"Jatuh! Sudut: {int(angle)}", (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        
        if frame_fall_status:
            fall_counter += 1
        else:
            fall_counter = 0
            
        # JIKA TERDETEKSI JATUH VALID
        if fall_counter >= FALL_TIME_THRESHOLD:
            is_fall_detected = True
            
            # --- SIMPAN BUKTI GAMBAR DI SINI ---
            save_path = os.path.join(OUTPUT_IMAGE_DIR, f"Bukti_{filename}.jpg")
            cv2.imwrite(save_path, evidence_frame)
            # -----------------------------------
            
            break 
            
    cap.release()
    return is_fall_detected

# --- MAIN PROGRAM ---
print("Memuat Model...")
model = YOLO(MODEL_PATH)
results_data = []

# 1. PROSES VIDEO JATUH
print(f"\n--- Memproses Folder FALLS ---")
if os.path.exists(PATH_FALLS):
    files = os.listdir(PATH_FALLS)
    for idx, filename in enumerate(files):
        if filename.endswith((".mp4", ".avi", ".mkv")):
            path = os.path.join(PATH_FALLS, filename)
            print(f"[{idx+1}/{len(files)}] {filename} ... ", end="")
            prediction = detect_fall_in_video(path, model)
            status = "TERDETEKSI (FOTO DISIMPAN)" if prediction else "GAGAL"
            print(f"-> {status}")
            results_data.append({"Filename": filename, "Actual": "Jatuh", "Pred": "Jatuh" if prediction else "Normal"})

# 2. PROSES VIDEO ADL
print(f"\n--- Memproses Folder ADL ---")
if os.path.exists(PATH_ADL):
    files = os.listdir(PATH_ADL)
    for idx, filename in enumerate(files):
        if filename.endswith((".mp4", ".avi", ".mkv")):
            path = os.path.join(PATH_ADL, filename)
            print(f"[{idx+1}/{len(files)}] {filename} ... ", end="")
            prediction = detect_fall_in_video(path, model)
            # Kalau di ADL terdeteksi jatuh, berarti False Alarm (tetap disimpan fotonya buat analisis)
            status = "SALAH DETEKSI (FOTO DISIMPAN)" if prediction else "AMAN"
            print(f"-> {status}")
            results_data.append({"Filename": filename, "Actual": "Normal", "Pred": "Jatuh" if prediction else "Normal"})

# --- HITUNG DAN SIMPAN ---
df = pd.DataFrame(results_data)
df.to_csv("Hasil_Uji_Akurasi.csv", index=False)
print("\nSelesai! Cek folder 'hasil_visual_uji' untuk melihat bukti foto.")