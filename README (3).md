# 🩻 Flask DICOM Classification --- README

Proyek ini adalah aplikasi Flask untuk memproses file **DICOM X-Ray**,
menampilkan hasil prediksi model deep learning, dan memberikan **tindak
lanjut otomatis** berdasarkan kelas prediksi. Aplikasi mendukung
**prediksi single**, **multiple**, menampilkan **visualisasi**, dan
menghasilkan **CSV** hasil klasifikasi.

## 1. 🔧 Setup Lingkungan

### 1.1 Install Python

Pastikan Python **≥ 3.8** sudah terpasang.

### 1.2 Buat Virtual Environment

``` bash
python -m venv venv
```

Aktifkan environment:

**Windows**

``` bash
venv\Scripts\activate
```

**Linux & MacOS**

``` bash
source venv/bin/activate
```

Masuk ke folder proyek:

``` bash
cd E:\AI2025\flask
```

### 1.3 Install Dependensi

``` bash
pip install -r requirements.txt
pip install Flask flask-cors
```

## 2. 🚀 Hello World Flask

``` python
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/")
def hello():
    return jsonify({"message": "Hello, Flask!"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
```

## 3. 📁 Struktur Project

    your_project/
    │
    ├── static/
    │   └── style.css
    ├── templates/
    │   └── index.html
    ├── app.py
    ├── best_model.keras
    └── requirements.txt

## 4. ▶️ Menjalankan Project

``` bash
python.exe -m pip install --upgrade pip
pip install flask tensorflow pydicom opencv-python pillow flask-cors numpy
python app.py
```

Akses: http://localhost:5000

## 5. 🖥️ Instalasi TensorFlow

### Raspberry Pi

Download wheel ARM: tensorflow-2.11.0-cp39-none-linux_aarch64.whl

### MacOS M1/M2/M3

``` bash
pip install tensorflow-macos tensorflow-metal
```

### Windows

``` bash
pip install tensorflow
```

## 6. 📊 Fitur Tindak Lanjut

Aplikasi memberikan informasi: - Pengulangan - Deskripsi - Tindak Lanjut

## 7. 📄 Export CSV

Saat multiple upload, sistem: - Menghitung jumlah kelas - Membuat tabel
rangkuman - Export CSV otomatis - Menandai file gagal diproses

## 8. 📘 Lisensi

MIT License
