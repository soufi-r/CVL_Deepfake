# 🕵️‍♂️ Deepfake Video Detection using Hybrid CNN + Bi-LSTM

![Deepfake Detection Banner](https://img.shields.io/badge/Deepfake-Detection-red?style=for-the-badge&logo=security)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)

> **Implementasi arsitektur Hybrid Convolutional Neural Network (CNN) dan Bidirectional Long Short-Term Memory (Bi-LSTM) untuk mendeteksi video manipulasi wajah (Deepfake) pada dataset UADFV.**

---

## Resource
- Link Video: [Youtube](https://www.youtube.com/watch?v=s-pvREP7Bp4)
- Link Paper: [Paper](https://drive.google.com/file/d/1QwaAVYkWJlOastN8TKKTDyIu-KrX8VNg/view?usp=sharing)

## 📖 Pendahuluan

Penelitian ini bertujuan untuk menerapkan dan mengevaluasi performa arsitektur **Hybrid CNN + Bidirectional LSTM (Bi-LSTM)** dalam tugas deteksi video *deepfake* menggunakan dataset **UADFV**. Fokus utama penelitian adalah klasifikasi biner (video asli vs. palsu) melalui proses *preprocessing* video yang sistematis dan ekstraksi fitur spatiotemporal.

Eksperimen ini dilakukan untuk menjawab kebutuhan akan metode deteksi *deepfake* yang lebih akurat dan *robust*, mengingat meningkatnya ancaman penyalahgunaan teknologi generatif (GANs) untuk manipulasi wajah. Penelitian ini berupaya meningkatkan hasil studi terdahulu dengan menggabungkan kemampuan ekstraksi fitur visual dari CNN dan pemahaman konteks temporal dari Bi-LSTM.

---

## 📚 Landasan Teori

Ancaman teknologi *deepfake* terhadap integritas informasi digital telah mendorong pengembangan metode deteksi yang canggih. Penelitian awal oleh **Yang et al. (2019)** berfokus pada inkonsistensi fisik seperti pose kepala (*head poses*). Namun, seiring evolusi teknik manipulasi yang semakin halus, pendekatan *hybrid* yang menggabungkan analisis spasial dan temporal menjadi standar baru.

Dalam arsitektur ini:
* **CNN (Convolutional Neural Network):** Berfungsi sebagai pengekstraksi fitur spasial yang kuat dari setiap frame video. Kami menggunakan varian **EfficientNetV2B0** yang telah terbukti efektif dalam menangkap detail visual mikro.
* **Bi-LSTM (Bidirectional LSTM):** Digunakan untuk memproses urutan fitur antar-frame guna mendeteksi inkonsistensi temporal (seperti kedipan mata yang tidak wajar atau *jitter* pada wajah) yang sering terjadi pada video palsu.

---

## 🔬 Metodologi Riset

### Dataset
Penelitian ini menggunakan dataset **UADFV (UAlbany DeepFake Video)** yang terdiri dari total **98 video**, terbagi rata menjadi dua kelas:
* **49 Video Asli (Real)**
* **49 Video Palsu (Fake)**

### Preprocessing
Tahapan pra-pemrosesan data dilakukan sebagai berikut:
1.  **Frame Extraction:** Mengekstrak setiap video menjadi **30 frame** berurutan menggunakan OpenCV.
2.  **Face Detection:** Mendeteksi dan memotong (*crop*) area wajah (ROI) pada setiap frame menggunakan **Haar Cascade**.
3.  **Resizing:** Mengubah ukuran citra wajah menjadi **96x96 piksel**.
4.  **Data Splitting:** Membagi dataset dengan rasio **70% Training**, **30% Testing**.
5.  **Augmentation:** Menerapkan augmentasi data pada *training set* (Flip, Rotasi, Zoom, Contrast) untuk mencegah *overfitting*.

### Arsitektur Model
Model yang diusulkan menggabungkan **CNN (EfficientNetV2B0)** sebagai *feature extractor* dan **Bi-LSTM** sebagai *sequence classifier*.

| Layer / Parameter | Konfigurasi |
| :--- | :--- |
| **Input Shape** | `(None, 30, 96, 96, 3)` |
| **Feature Extractor** | `TimeDistributed(EfficientNetV2B0)` |
| **CNN Output** | `(None, 30, 3, 3, 1280)` |
| **Flatten Layer** | `TimeDistributed(Flatten)` → Output: `(None, 30, 11520)` |
| **Bi-LSTM 1** | `Bidirectional(LSTM, 128 units)` → Output: `(None, 30, 256)` |
| **Bi-LSTM 2** | `Bidirectional(LSTM, 64 units)` → Output: `(None, 128)` |
| **Dense Layers** | 128 units (ReLU) → 64 units (ReLU) |
| **Regularization** | Dropout (0.2 - 0.6) & Batch Normalization |
| **Output Layer** | `Dense(1, activation="sigmoid")` |
| **Optimizer** | Adam (Learning Rate: 0.0001) |
| **Loss Function** | Binary Crossentropy |
| **Total Params** | **18,038,609** |

---

## 📊 Hasil Eksperimen

Model yang diusulkan berhasil mencapai performa yang kompetitif pada dataset UADFV.

### Confusion Matrix
Berikut adalah hasil klasifikasi pada data uji:


![Confusion Matrix](https://github.com/soufi-r/CVL_Deepfake/blob/master/confusion_ROC.jpg?raw=true)
> *Gambar 1: Confusion Matrix menunjukkan model mampu membedakan video asli dan palsu dengan tingkat kesalahan yang minim.*

### ROC Curve
Kurva ROC menunjukkan nilai **AUC sebesar 0.898**, yang mengindikasikan kemampuan diskriminasi model yang sangat baik.


### Perbandingan Performa
| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Proposed (Hybrid)** | **0.83** | **0.85** | **0.83** | **0.83** |
| SVM (Baseline) | - | - | - | - |

---

## 🏁 Kesimpulan

Penelitian ini berhasil mengimplementasikan arsitektur *Hybrid* CNN dan Bi-LSTM untuk deteksi *deepfake*. Integrasi **EfficientNet** untuk ekstraksi fitur spasial detail dan **Bi-LSTM** untuk analisis temporal terbukti efektif dalam menangkap anomali pada video manipulasi.

Hasil evaluasi menunjukkan bahwa model mampu mencapai nilai **AUC 0.898**, mengungguli metode konvensional berbasis fitur fisik (seperti inkonsistensi pose kepala). Hal ini menegaskan bahwa pendekatan *spatio-temporal* adalah solusi yang menjanjikan untuk menghadapi ancaman *deepfake* yang semakin canggih.

---

## 👥 Kontributor

Terima kasih kepada seluruh anggota tim yang telah berkontribusi dalam penelitian ini:

- 👨‍💻 **Bintang M. M.** (@abinmadani)
- 👨‍💻 **Dzaky N. A.** (@Lacoshh)
- 👩‍💻 **Soufi R. I.** (@soufi-r)
- 👨‍💻 **Muhamad A. C. F.** (@cholilfayyadl)

---
*Dibuat untuk memenuhi tugas Project Akhir Mata Kuliah Computer Vision.*
