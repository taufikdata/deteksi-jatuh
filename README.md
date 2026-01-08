# Sistem Deteksi Jatuh (Fall Detection System) - YOLOv8 Pose

Proyek ini menggunakan logika Hybrid Rule-Based (Platinum V34) untuk mendeteksi manusia jatuh pada video CCTV pendek.
Akurasi saat ini: ~88% (Tested on 100 Dataset).

## 📂 Struktur Folder
- `code_V34.py`: Script utama (Main Program).
- `dataset/`: Folder berisi 100 video (50 Falls, 50 ADL).
- `requirements.txt`: Daftar library yang dibutuhkan.

## 🚀 Cara Instalasi (Untuk Anggota Kelompok)

**Langkah 1: Clone atau Download Repo ini**

**Langkah 2: Siapkan Virtual Environment (Wajib)**
Buka terminal di folder ini, lalu ketik:
```bash
# Untuk Windows
python -m venv venv
venv\Scripts\activate

# Untuk Mac/Linux
python3 -m venv venv
source venv/bin/activate