# Pendahuluan
Penelitian ini bertujuan untuk menerapkan dan mengevaluasi performa arsitektur CNN + Bidirectional LSTM (BiLSTM) dalam tugas deteksi video deepfake menggunakan dataset UADFV. Fokus penelitian berada pada klasifikasi dua kelas, yaitu asli dan palsu, melalui proses preprocessing dan ekstraksi fitur dari frame video.
Eksperimen ini dilakukan untuk menjawab kebutuhan akan metode deteksi deepfake yang lebih akurat, mengingat meningkatnya penyalahgunaan teknologi generatif dalam bentuk manipulasi wajah dan video. Penelitian ini juga bertujuan meningkatkan hasil dari studi sebelumnya dengan memanfaatkan kombinasi CNN dan BiLSTM, serta mengevaluasi performa model menggunakan metrik accuracy, precision, recall, dan F1-score.
Hasil penelitian diharapkan dapat memberikan kontribusi dalam pengembangan sistem deteksi deepfake yang lebih efektif pada konteks cybersecurity dan digital forensics, serta menjadi referensi bagi penelitian lanjutan terkait deteksi video palsu.

# Landasan Teori
....

# Metodologi Riset
Metodologi pada penelitian ini adalah sebagai berikut 

## Dataset
Pada penelitian ini, dataset yang digunakan berupa dataset UADV, yang terdiri dari 98 video yang dubedakan menjadi 2 kelas yaitu kelas asli dan kelas fake

## Preprocessing
1. Mengekstrak setiap video menjadi 30 frame menggunakan openCV. 
2. Mendeteksi wajah dari setiap frame yang telah didapatkan menggunakan haar cascade
3. Memotong bagian wajah dan mengganti ukuran menjadi 96x96
4. Membagi dataset menjadi 3 dengan ukuran sebanyak 60% dari keseluruhan data digunakan sebagai training, sebanyak 20% keseluruhan data digunakan untuk validasi dan sisanya akan digunakan sebagai testing.
5. Melakukan augmentasi pada data training dengan cara flip, rotasi, zoom, dan contrast

## Model
Model yang kami gunakan berupa CNN+BiLSTM, dimana metode CNN akan digunakan untuk mengekstraksi fitur dari setiap frame citra dan bilstm akan digunakan untuk proses pembelajaran dan klasifikasi dengan memperhatikan pergerakan yang tidak natural dan
tidak konsisten.

### Arsitektur Model
| Layer / Parameter        | Konfigurasi                                                 |
| ------------------------ | ----------------------------------------------------------- |
| Input Shape              | `(None, 30, 96, 96, 3)`                                     |
| Feature Extractor        | `TimeDistributed(ConvNet / CNN extractor)`                  |
| CNN Output Shape         | `(None, 30, 3, 3, 1280)`                                    |
| Flatten Layer            | `TimeDistributed(Flatten)` → output `(None, 30, 11520)`     |
| BiLSTM Layer 1           | `Bidirectional(LSTM, units=128)` → output `(None, 30, 256)` |
| Batch Normalization 1    | `(None, 30, 256)`                                           |
| BiLSTM Layer 2           | `Bidirectional(LSTM, units=64)` → output `(None, 128)`      |
| Batch Normalization 2    | `(None, 128)`                                               |
| Dense Layer 1            | `Dense(128)`                                                |
| Dropout 1                | `Dropout(0.2)`                                              |
| Dense Layer 2            | `Dense(64)`                                                 |
| Dropout 2                | `Dropout(0.2)`                                              |
| Output Layer             | `Dense(1, activation="sigmoid")`                            |
| Optimizer                | Adam                                                        |
| Loss Function            | Binary Crossentropy                                         |
| Total Parameters         | **18,038,609**                                              |
| Trainable Parameters     | **12,118,529**                                              |
| Non-trainable Parameters | **5,920,080**                                               |


# Hasil
Hasil uji model menggunakan data uji ditampilkan pada confusion matrix dan ROC 
...

# Kesimpulan
....

# Contributor
@abinmadani
@Lacoshh
@soufi-r
@...
