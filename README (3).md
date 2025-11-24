### 1. 🔧 Setup Lingkungan
#buka terminal
•	Install Python (pastikan sudah di atas 3.8)
•	Buat virtual environment:
```bash
python -m venv venv #copy paste pada terminal
#source venv/bin/activate  # atau 
venv\Scripts\activate #di Windows copy paste di terminal
```
cd E:\AI2025\flask      #copas di terminal

pip install -r requirements.txt         #copas di terminal

jika tidak bisa di jalankan maka : Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass


•	Install Flask dan SQLite support:
```bash
pip install Flask flask-cors      #coopas di terminal
```
### 2. Membuat Hello World & Response Object

```py     #copas di terminal
from flask import Flask, jsonify      # ditulis di terminal
from flask_cors import CORS          #ditulis di terminal
exit ( )                            #ditulis di terminal
python app.py                       #copas di terminal
#copy url buka di chrome


app = Flask(__name__)
# Mengaktifkan CORS untuk seluruh aplikasi
CORS(app)

@app.route("/")
def hepython -m venv venvllo():
    return jsonify({"message": "Hello, Flask!"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
```


### 3. Struktur Project


your_project/
│
├── static/
│   └── style.css        ← (opsional untuk custom style)
│
├── templates/
│   └── index.html       ← frontend HTML
│
├── app.py               ← Flask backend
├── best_model.keras     ← model hasil training kamu
└── requirements.txt     ← tensorflow, flask, pydicom, opencv-python, numpy


```py
#app.py 
from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import pydicom
import tensorflow as tf
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
model = tf.keras.models.load_model("best_model.keras")
classes = ['ACCEPT', 'ARTEFAK', 'EXPOSURE', 'POSITION']

def read_dicom_as_rgb(dicom_bytes):
    dicom = pydicom.dcmread(BytesIO(dicom_bytes))
    image = dicom.pixel_array
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image

def preprocess_image(image):
    image_resized = cv2.resize(image, (256, 256))
    return image_resized / 255.0

def encode_image_to_base64(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(np.uint8(image_rgb))
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    dicom_bytes = file.read()

    image = read_dicom_as_rgb(dicom_bytes)
    processed = preprocess_image(image)
    input_tensor = np.expand_dims(processed, axis=0)

    preds = model.predict(input_tensor)[0]
    predicted_index = np.argmax(preds)
    predicted_class = classes[predicted_index]

    base64_image = encode_image_to_base64(image)

    probs = sorted(
    [{"class_name": classes[i], "percent": f"{preds[i]*100:.2f}"} for i in range(len(classes))],
    key=lambda x: float(x["percent"]),
    reverse=True
    )
    
    return jsonify({
        "predicted_class": predicted_class,
        "probabilities": probs,
        "image": base64_image
    })

@app.route('/predict-multiple', methods=['POST'])
def predict_multiple():
    files = request.files.getlist('files')
    if not files:
        return jsonify({"error": "No files uploaded."}), 400

    result_counter = {cls: 0 for cls in classes}

    for file in files:
        dicom_bytes = file.read()
        try:
            image = read_dicom_as_rgb(dicom_bytes)
            processed = preprocess_image(image)
            input_tensor = np.expand_dims(processed, axis=0)
            preds = model.predict(input_tensor)[0]
            predicted_index = np.argmax(preds)
            predicted_class = classes[predicted_index]
            result_counter[predicted_class] += 1
        except Exception as e:
            print(f"Failed to process {file.filename}: {e}")

    total_files = len(files)
    result_summary = []
    for cls in classes:
        count = result_counter[cls]
        percent = (count / total_files) * 100 if total_files else 0
        result_summary.append({
            "class_name": cls,
            "count": count,
            "percent": f"{percent:.2f}"
        })

    result_summary.sort(key=lambda x: x["count"], reverse=True)

    return jsonify({
        "total_files": total_files,
        "summary": result_summary
    })


if __name__ == "__main__":
    app.run(debug=True)

```

templates/index.html

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Prediksi Gambar X-Ray</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="container py-4">

  <h2 class="mb-4">Upload Gambar DICOM untuk Prediksi</h2>

  <form id="upload-form" enctype="multipart/form-data">
    <input class="form-control mb-3" type="file" name="file" accept=".dcm" required>
    <button class="btn btn-primary" type="submit">Prediksi</button>
  </form>

  <div id="preview" class="mt-4">
    <img id="dicom-image" class="img-fluid mb-3" style="max-height: 300px;" />
    <div id="result" class="alert alert-info d-none"></div>
  </div>

  <script>
    const form = document.getElementById("upload-form");
    const resultDiv = document.getElementById("result");
    const imgTag = document.getElementById("dicom-image");

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const formData = new FormData(form);

      const res = await fetch("/predict", {
        method: "POST",
        body: formData
      });
      const data = await res.json();

      // Tampilkan gambar
      imgTag.src = "data:image/png;base64," + data.image;

      // Tampilkan hasil prediksi
      resultDiv.classList.remove("d-none");
      resultDiv.innerHTML = `<strong>Prediksi:</strong> ${data.predicted_class}<br><br>` +
        data.probabilities.map(p =>
          `${p.class_name}: ${p.percent}%`
        ).join("<br>");
    });
  </script>

</body>
</html>

```

3. Jalankan Project
bash
python.exe -m pip install --upgrade pip
pip install flask tensorflow pydicom opencv-python pillow flask-cors numpy
python app.py
Akses via browser:
http://localhost:5000


pip install -r requirements.txt

untuk raspberry : https://github.com/lhelontra/tensorflow-on-arm/releases/download/v2.11.0/tensorflow-2.11.0-cp39-none-linux_aarch64.whl
untuk macos : tensorflow
untuk windows : tensorflow

### 4. Lengkapi dengan tindak lanjut 

```py
#app.py
from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import pydicom
import tensorflow as tf
import base64
from io import BytesIO
from PIL import Image
from flask_cors import CORS
import csv
import io

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model("best_model.keras")
classes = ['ACCEPT', 'ARTEFAK', 'EXPOSURE', 'POSITION']

# Data tindak lanjut
tindaklanjut = [
    {
        "dataid": "POSITION",
        "pengulangan": [
            "Mengalami rotasi/ posisi miring",
            "Pasien salah dalam orientasi"
        ],
        "diskripsi": [
            "Posisi anatomi yang tidak tepat",
            "Posisi tidak true, mengalami rotasi baik ke internal maupun eksternal"
        ],
        "tindaklanjut": [
            "Memastikan semua radiografer memiliki sertifikasi kompetensi",
            "Adopsi protokol standar posisi pasien dan teknik citra",
            "Menyediakan panduan prosedur tertulis",
            "Membangun komunikasi efektif dengan pasien",
            "Audit rutin citra radiografi untuk pola kesalahan"
        ]
    },
    {
        "dataid": "ARTEFAK",
        "pengulangan": [
            "Obyek yang dikenali",
            "Grid line atau artefak sejenis",
            "Ketidakseragaman atau defect/cacat terlihat"
        ],
        "diskripsi": [
            "Obyek seperti kancing, perhiasan, shield",
            "Artefak interferensi elektromagnetik",
            "Artefak detector seperti piksel mati"
        ],
        "tindaklanjut": [
            "Mengurangi gangguan elektromagnetik",
            "Memastikan pasien bebas dari bahan logam",
            "Pemeliharaan dan kalibrasi alat rutin",
            "Screening pasien sebelum pemeriksaan",
            "Audit rutin citra radiografi"
        ]
    },
    {
        "dataid": "EXPOSURE",
        "pengulangan": [
            "Under eksposure",
            "Over eksposure/saturasi"
        ],
        "diskripsi": [
            "Noise berlebih akibat eksposure kurang",
            "Over/under density akibat faktor teknik"
        ],
        "tindaklanjut": [
            "Adopsi protokol standar",
            "Pencatatan hasil & koreksi",
            "Komunikasi efektif antara radiografer-teknisi",
            "Pemeliharaan rutin alat",
            "Audit citra radiografi berkala"
        ]
    },
    {
        "dataid": "ACCEPT",
        "pengulangan": [],
        "diskripsi": [],
        "tindaklanjut": []
    },
]

def read_dicom_as_rgb(dicom_bytes):
    dicom = pydicom.dcmread(BytesIO(dicom_bytes))
    image = dicom.pixel_array
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image

def preprocess_image(image):
    image_resized = cv2.resize(image, (256, 256))
    return image_resized / 255.0

def encode_image_to_base64(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(np.uint8(image_rgb))
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    dicom_bytes = file.read()

    image = read_dicom_as_rgb(dicom_bytes)
    processed = preprocess_image(image)
    input_tensor = np.expand_dims(processed, axis=0)

    preds = model.predict(input_tensor)[0]
    predicted_index = np.argmax(preds)
    predicted_class = classes[predicted_index]

    base64_image = encode_image_to_base64(image)

    probs = sorted(
        [{"class_name": classes[i], "percent": f"{preds[i]*100:.2f}"} for i in range(len(classes))],
        key=lambda x: float(x["percent"]),
        reverse=True
    )

    matched_action = next((item for item in tindaklanjut if item["dataid"] == predicted_class), None)

    return jsonify({
        "predicted_class": predicted_class,
        "probabilities": probs,
        "image": base64_image,
        "action": matched_action
    })

@app.route('/predict-multiple', methods=['POST'])
def predict_multiple():
    files = request.files.getlist('files')
    if not files:
        return jsonify({"error": "No files uploaded."}), 400

    result_counter = {cls: 0 for cls in classes}
    csv_rows = []
    error_files = []

    for idx, file in enumerate(files, 1):
        filename = file.filename
        dicom_bytes = file.read()
        try:
            # Proses file DICOM
            image = read_dicom_as_rgb(dicom_bytes)
            processed = preprocess_image(image)
            input_tensor = np.expand_dims(processed, axis=0)

            preds = model.predict(input_tensor)[0]
            predicted_index = np.argmax(preds)
            predicted_class = classes[predicted_index]

            result_counter[predicted_class] += 1
            csv_rows.append([idx, filename, predicted_class])
        except Exception as e:
            print(f"Failed to process {filename}: {e}")
            error_files.append(filename)
            csv_rows.append([idx, filename, "ERROR"])

    total_files = len(files)
    result_summary = []
    for cls in classes:
        count = result_counter[cls]
        percent = (count / total_files) * 100 if total_files else 0
        result_summary.append({
            "class_name": cls,
            "count": count,
            "percent": f"{percent:.2f}",
            
        })

    result_summary.sort(key=lambda x: x["count"], reverse=True)
    
    matched_action = next((item for item in tindaklanjut if item["dataid"] == predicted_class), None)

    # Generate CSV di memory
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["No", "Nama File", "Prediksi Kelas Tertinggi"])
    writer.writerows(csv_rows)
    csv_data = csv_buffer.getvalue()

    response_data = {
        "total_files": total_files,
        "summary": result_summary,
        "csv": csv_data,
        "action": matched_action
    }

    if error_files:
        response_data["errors"] = {
            "count": len(error_files),
            "files": error_files
        }

    return jsonify(response_data)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

```

index.html

```html
<!DOCTYPE html>
<html lang="en">

<head>
  <meta charset="UTF-8">
  <title>Prediksi Gambar X-Ray</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>

<body class="container py-4">

  <h2 class="mb-4">Upload Gambar DICOM untuk Prediksi</h2>

  <!-- Tombol Refresh -->
  <button id="refresh-button" class="btn btn-outline-danger mb-4">Refresh Prediksi</button>

  <!-- Form Prediksi Single -->
  <form id="upload-form" enctype="multipart/form-data" class="mb-4">
    <input class="form-control mb-2" type="file" name="file" accept=".dcm" required>
    <button class="btn btn-primary" type="submit">Prediksi Single</button>
  </form>

  <!-- Form Prediksi Multiple -->
  <form id="multi-form" enctype="multipart/form-data" class="mb-5">
    <input class="form-control mb-2" type="file" name="files" accept=".dcm" multiple required>
    <button class="btn btn-success" type="submit">Prediksi Multiple</button>
  </form>

  <div id="loading" class="alert alert-warning d-none">Memproses, mohon tunggu...</div>

  <!-- Preview hasil Single -->
  <div id="preview" class="mb-4">
    <img id="dicom-image" class="img-fluid mb-3" style="max-height: 300px;" />
    <div id="result" class="alert alert-info d-none"></div>
  </div>

  <!-- Card 3 bagian untuk Tindak Lanjut -->
  <div id="action-result" class="row d-none">
    <div class="col-md-4 mb-3">
      <div class="card h-100">
        <div class="card-header bg-primary text-white">Pengulangan</div>
        <div class="card-body">
          <ul id="pengulangan-list" class="list-group list-group-flush"></ul>
        </div>
      </div>
    </div>
    <div class="col-md-4 mb-3">
      <div class="card h-100">
        <div class="card-header bg-warning text-dark">Deskripsi</div>
        <div class="card-body">
          <ul id="diskripsi-list" class="list-group list-group-flush"></ul>
        </div>
      </div>
    </div>
    <div class="col-md-4 mb-3">
      <div class="card h-100">
        <div class="card-header bg-success text-white">Tindak Lanjut</div>
        <div class="card-body">
          <ul id="tindaklanjut-list" class="list-group list-group-flush"></ul>
        </div>
      </div>
    </div>
  </div>

  <!-- Hasil Multiple -->
  <div id="multi-result" class="d-none">
    <h4>Hasil Prediksi Multiple</h4>
    <p id="summary-text"></p>
    <ul id="class-summary" class="list-group mb-3"></ul>
    <a id="download-csv" class="btn btn-outline-primary" download="hasil_prediksi.csv">Download CSV</a>
  </div>

  <script>
    const uploadForm = document.getElementById("upload-form");
    const multiForm = document.getElementById("multi-form");
    const resultDiv = document.getElementById("result");
    const imgTag = document.getElementById("dicom-image");
    const loadingDiv = document.getElementById("loading");

    const actionResult = document.getElementById("action-result");
    const pengulanganList = document.getElementById("pengulangan-list");
    const diskripsiList = document.getElementById("diskripsi-list");
    const tindaklanjutList = document.getElementById("tindaklanjut-list");

    const multiResult = document.getElementById("multi-result");
    const summaryText = document.getElementById("summary-text");
    const classSummary = document.getElementById("class-summary");
    const downloadCSV = document.getElementById("download-csv");

    // Handle upload single
    uploadForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const formData = new FormData(uploadForm);
      loadingDiv.classList.remove("d-none");

      const res = await fetch("/predict", { method: "POST", body: formData });
      const data = await res.json();
      loadingDiv.classList.add("d-none");

      imgTag.src = "data:image/png;base64," + data.image;
      resultDiv.classList.remove("d-none");
      resultDiv.innerHTML = `<strong>Prediksi:</strong> ${data.predicted_class}<br><br>` +
        data.probabilities.map(p => `${p.class_name}: ${p.percent}%`).join("<br>");

      if (data.action) {
        actionResult.classList.remove("d-none");

        pengulanganList.innerHTML = data.action.pengulangan.length > 0
          ? data.action.pengulangan.map(item => `<li class="list-group-item">${item}</li>`).join("")
          : '<li class="list-group-item">-</li>';

        diskripsiList.innerHTML = data.action.diskripsi.length > 0
          ? data.action.diskripsi.map(item => `<li class="list-group-item">${item}</li>`).join("")
          : '<li class="list-group-item">-</li>';

        tindaklanjutList.innerHTML = data.action.tindaklanjut.length > 0
          ? data.action.tindaklanjut.map(item => `<li class="list-group-item">${item}</li>`).join("")
          : '<li class="list-group-item">-</li>';
      } else {
        actionResult.classList.add("d-none");
      }
    });

    // Handle upload multiple
    multiForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const formData = new FormData(multiForm);
      loadingDiv.classList.remove("d-none");

      const res = await fetch("/predict-multiple", { method: "POST", body: formData });
      const data = await res.json();
      loadingDiv.classList.add("d-none");
      multiResult.classList.remove("d-none");

      summaryText.innerText = `Jumlah File: ${data.total_files} File`;
      classSummary.innerHTML = data.summary.map((cls, idx) =>
        `<li class="list-group-item">${idx + 1}. ${cls.class_name}: ${cls.count} File (${cls.percent}%)`
      ).join("");

      const blob = new Blob([data.csv], { type: "text/csv" });
      const url = URL.createObjectURL(blob);
      downloadCSV.href = url;

      if (data.action) {
        actionResult.classList.remove("d-none");

        pengulanganList.innerHTML = data.action.pengulangan.length > 0
          ? data.action.pengulangan.map(item => `<li class="list-group-item">${item}</li>`).join("")
          : '<li class="list-group-item">-</li>';

        diskripsiList.innerHTML = data.action.diskripsi.length > 0
          ? data.action.diskripsi.map(item => `<li class="list-group-item">${item}</li>`).join("")
          : '<li class="list-group-item">-</li>';

        tindaklanjutList.innerHTML = data.action.tindaklanjut.length > 0
          ? data.action.tindaklanjut.map(item => `<li class="list-group-item">${item}</li>`).join("")
          : '<li class="list-group-item">-</li>';
      } else {
        actionResult.classList.add("d-none");
      }

    });

    const refreshButton = document.getElementById("refresh-button");

    refreshButton.addEventListener("click", () => {
      location.reload(); // Refresh seluruh halaman
    });

  </script>

</body>

</html>
```
