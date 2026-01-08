import cv2
import os
import pandas as pd
import math
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve, average_precision_score

# ==========================================
# 1. KONFIGURASI
# ==========================================
PATH_FALLS = 'dataset/falls' 
PATH_ADL   = 'dataset/adl'
OUTPUT_FOLDER = 'Laporan_Final_V34_Visual' # Saya ganti nama foldernya biar fresh

if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)
OUTPUT_BUKTI = os.path.join(OUTPUT_FOLDER, 'bukti_visual')
if not os.path.exists(OUTPUT_BUKTI): os.makedirs(OUTPUT_BUKTI)

# --- PARAMETER DETEKSI (V19-FIX) ---
CONF_THRESHOLD = 0.5
SPEED_CHECK_FRAMES = 10
SPEED_THRESHOLD = 80
FLOOR_LIMIT_LOW = 0.35
ANGLE_LOOSE = 65
RATIO_LOOSE = 0.8
FLOOR_LIMIT_HIGH = 0.6
ANGLE_STRICT = 45
RATIO_STRICT = 1.2
INIT_CHECK_FRAMES = 6
INIT_LYING_ANGLE = 45
INIT_LYING_RATIO = 1.15

# ==========================================
# 2. LOGIKA DETEKSI
# ==========================================
def calculate_angle(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    if x2 - x1 == 0: return 90.0
    slope = (y2 - y1) / (x2 - x1)
    return abs(math.degrees(math.atan(slope)))

def analyze_video(video_path, model, save_proof=True):
    cap = cv2.VideoCapture(video_path)
    filename = os.path.basename(video_path)

    max_speed_recorded = 0
    y_history = deque(maxlen=SPEED_CHECK_FRAMES)
    flag_speed = False
    flag_pose = False
    
    init_frames_seen = 0
    sleep_hits = 0
    sleep_mode = False
    SLEEP_HITS_NEED = 4 
    
    detection_confidence = 0.0
    start_time = time.time()
    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success: break
        
        frame_count += 1
        h_img = frame.shape[0]
        w_img = frame.shape[1]
        y_floor = int(h_img * (1.0 - FLOOR_LIMIT_LOW))
        y_middle = int(h_img * (1.0 - FLOOR_LIMIT_HIGH))

        results = model(frame, conf=CONF_THRESHOLD, verbose=False)
        got_person = False

        for result in results:
            if result.keypoints is None: continue
            kps = result.keypoints.data.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()

            for i, person in enumerate(kps):
                if (person[5][2] < 0.5 or person[6][2] < 0.5 or
                    person[11][2] < 0.5 or person[12][2] < 0.5):
                    continue

                got_person = True
                sh_x = int((person[5][0] + person[6][0]) / 2)
                sh_y = int((person[5][1] + person[6][1]) / 2)
                hip_x = int((person[11][0] + person[12][0]) / 2)
                hip_y = int((person[11][1] + person[12][1]) / 2)

                y_history.append(sh_y)
                angle = calculate_angle((sh_x, sh_y), (hip_x, hip_y))
                
                box = boxes[i]
                w = box[2] - box[0]
                h = box[3] - box[1]
                ratio = w / h if h > 0 else 0

                # --- SLEEP CHECK ---
                if not sleep_mode and init_frames_seen < INIT_CHECK_FRAMES:
                    init_frames_seen += 1
                    is_lying = (angle < INIT_LYING_ANGLE) or (ratio > INIT_LYING_RATIO)
                    if is_lying and hip_y > y_middle:
                        sleep_hits += 1
                    if init_frames_seen >= INIT_CHECK_FRAMES and sleep_hits >= SLEEP_HITS_NEED:
                        sleep_mode = True
                        cap.release()
                        end_time = time.time()
                        fps = frame_count / (end_time - start_time) if (end_time - start_time) > 0 else 0
                        return False, max_speed_recorded, 0.1, fps

                # --- CEK SPEED ---
                if len(y_history) == SPEED_CHECK_FRAMES:
                    current_speed = abs(y_history[-1] - y_history[0])
                    if current_speed > max_speed_recorded:
                        max_speed_recorded = current_speed
                    if current_speed > SPEED_THRESHOLD:
                        flag_speed = True

                # --- CEK POSE ---
                if hip_y > y_floor:
                    if angle < ANGLE_LOOSE or ratio > RATIO_LOOSE:
                        flag_pose = True
                elif hip_y > y_middle:
                    if angle < ANGLE_STRICT or ratio > RATIO_STRICT:
                        flag_pose = True

        if not got_person: continue
        if sleep_mode: continue

        if flag_speed and flag_pose:
            # Hitung Confidence
            norm_speed = min(max_speed_recorded / 300, 1.0)
            norm_angle = max(0, (90 - angle) / 90)
            detection_confidence = (norm_speed * 0.5) + (norm_angle * 0.5)
            if detection_confidence > 0.95: detection_confidence = 0.99
            
            if save_proof:
                # === BAGIAN INI YANG BIKIN GARIS-GARIS MUNCUL ===
                vis_frame = results[0].plot() # Mengambil gambar hasil render YOLO
                
                # Tambah info manual kita
                cv2.putText(vis_frame, f"JATUH V34! Spd:{int(max_speed_recorded)}", 
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # Gambar garis lantai (Kuning)
                cv2.line(vis_frame, (0, y_floor), (w_img, y_floor), (0, 255, 255), 2)
                # Gambar garis tengah (Merah)
                cv2.line(vis_frame, (0, y_middle), (w_img, y_middle), (0, 0, 255), 2)
                
                save_path = os.path.join(OUTPUT_BUKTI, f"BUKTI_{filename}.jpg")
                cv2.imwrite(save_path, vis_frame)
            
            cap.release()
            end_time = time.time()
            fps = frame_count / (end_time - start_time) if (end_time - start_time) > 0 else 0
            return True, max_speed_recorded, detection_confidence, fps

    cap.release()
    end_time = time.time()
    fps = frame_count / (end_time - start_time) if (end_time - start_time) > 0 else 0
    return False, max_speed_recorded, 0.1, fps

# ==========================================
# 3. FUNGSI TESTING & REPORTING
# ==========================================

def run_full_evaluation():
    print("\n🚀 MEMULAI PENGUJIAN Final LENGKAP (V34 - VISUAL)...")
    model = YOLO('yolov8n-pose.pt')

    y_true = [] 
    y_pred = []
    y_scores = [] 
    fps_list = []
    results_data = []

    # 1. Dataset Falls
    print(f"\n📂 Memproses Data JATUH ({PATH_FALLS})...")
    if os.path.exists(PATH_FALLS):
        files = [f for f in os.listdir(PATH_FALLS) if f.endswith((".mp4", ".avi", ".mkv"))]
        for idx, f in enumerate(files):
            print(f"[{idx+1}/{len(files)}] {f} ... ", end="")
            detected, speed, conf, fps = analyze_video(os.path.join(PATH_FALLS, f), model)
            
            status = "✅ SUKSES" if detected else "❌ GAGAL"
            print(f"{status} (FPS: {fps:.1f})")
            
            y_true.append(1)
            y_pred.append(1 if detected else 0)
            y_scores.append(conf)
            fps_list.append(fps)
            results_data.append({"File": f, "Jenis": "Jatuh", "Prediksi": "Jatuh" if detected else "Normal", "Benar": detected == True})

    # 2. Dataset ADL
    print(f"\n📂 Memproses Data NORMAL ({PATH_ADL})...")
    if os.path.exists(PATH_ADL):
        files = [f for f in os.listdir(PATH_ADL) if f.endswith((".mp4", ".avi", ".mkv"))]
        for idx, f in enumerate(files):
            print(f"[{idx+1}/{len(files)}] {f} ... ", end="")
            detected, speed, conf, fps = analyze_video(os.path.join(PATH_ADL, f), model)
            
            status = "✅ AMAN" if not detected else "⚠️ FALSE"
            print(f"{status} (FPS: {fps:.1f})")
            
            y_true.append(0)
            y_pred.append(1 if detected else 0)
            y_scores.append(conf)
            fps_list.append(fps)
            results_data.append({"File": f, "Jenis": "Normal", "Prediksi": "Jatuh" if detected else "Normal", "Benar": detected == False})

    # 3. Generate Laporan
    print("\n📊 MENGHASILKAN GRAFIK & DATA Final...")
    
    # Save CSV
    df = pd.DataFrame(results_data)
    df.to_csv(os.path.join(OUTPUT_FOLDER, "Laporan_Data_V34.csv"), index=False)

    # Metrics
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0

    print("-" * 30)
    print(f"HASIL AKHIR PLATINUM V34")
    print("-" * 30)
    print(f"Akurasi  : {acc*100:.2f}%")
    print(f"Presisi  : {prec*100:.2f}%")
    print(f"Recall   : {rec*100:.2f}%")
    print(f"F1-Score : {f1*100:.2f}%")
    print(f"Rata-rata FPS : {avg_fps:.2f}")
    print("-" * 30)

    # Grafik 1: Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Prediksi Normal', 'Prediksi Jatuh'], yticklabels=['Asli Normal', 'Asli Jatuh'])
    plt.title('Confusion Matrix')
    plt.ylabel('Kondisi Sebenarnya')
    plt.xlabel('Hasil Prediksi')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, '1_Confusion_Matrix.png'))

    # Grafik 2: Metrik Performa
    plt.figure(figsize=(8, 5))
    metrics = {'Akurasi': acc, 'Presisi': prec, 'Recall': rec, 'F1-Score': f1}
    bars = plt.bar(metrics.keys(), [v*100 for v in metrics.values()], color=['#4CAF50', '#2196F3', '#FF9800', '#F44336'])
    plt.ylim(0, 110)
    plt.title('Metrik Performa Sistem')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}%", ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, '2_Metrik_Performa.png'))

    # Grafik 3: ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, '3_Kurva_ROC.png'))

    # Grafik 4: Precision-Recall Curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)
    plt.figure(figsize=(7, 6))
    plt.plot(recall_curve, precision_curve, color='purple', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, '4_Kurva_Precision_Recall.png'))
    
    print(f"\n🎉 SELESAI! Laporan lengkap ada di folder '{OUTPUT_FOLDER}'")

def test_single_new_video(video_path):
    print(f"\n🎥 MENGUJI VIDEO BARU: {video_path}")
    if not os.path.exists(video_path):
        print("❌ File video tidak ditemukan!")
        return

    model = YOLO('yolov8n-pose.pt')
    detected, speed, conf, fps = analyze_video(video_path, model, save_proof=True)
    
    print("-" * 30)
    if detected:
        print(f"⚠️ HASIL: TERDETEKSI JATUH!")
        print(f"   - Max Speed: {speed}")
        print(f"   - Confidence: {conf:.2f}")
    else:
        print(f"✅ HASIL: Aktivitas Normal (Tidak Jatuh)")
        print(f"   - Max Speed: {speed}")
    print(f"   - Kecepatan Proses: {fps:.1f} FPS")
    print(f"   - Bukti tersimpan di folder '{OUTPUT_BUKTI}'")
    print("-" * 30)

# ==========================================
# MAIN MENU
# ==========================================
if __name__ == "__main__":
    print("Pilih Mode:")
    print("1. Generate Laporan Final (100 Data Dataset)")
    print("2. Test Video Baru (Data 101, 102, dll)")
    
    pilihan = input("Masukkan pilihan (1/2): ")
    
    if pilihan == '1':
        run_full_evaluation()
    elif pilihan == '2':
        video_name = input("Masukkan nama file video (contoh: video_baru.mp4): ")
        test_single_new_video(video_name)
    else:
        print("Pilihan tidak valid.")