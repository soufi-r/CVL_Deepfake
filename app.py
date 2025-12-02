import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import tensorflow as pd
from tensorflow.keras.models import load_model

# --- KONFIGURASI ---
MODEL_PATH = r'C:\Users\KuePuki\Documents\MKA\CVL\best_model_fold_5.keras'  # Pastikan nama file model sesuai
IMG_SIZE = 96
SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 0.5

# --- SETUP HALAMAN STREAMLIT ---
st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🕵️",
    layout="centered"
)

# --- CSS CUSTOM (Opsional: Agar tampilan lebih cantik) ---
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
    }
    .reportview-container {
        background: #f0f2f6
    }
    </style>
    """, unsafe_allow_html=True)

# --- FUNGSI LOAD MODEL (CACHE) ---
# @st.cache_resource agar model hanya di-load sekali saja di memori
@st.cache_resource
def load_trained_model():
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model. Pastikan file '{MODEL_PATH}' ada.")
        return None

# --- FUNGSI PREPROCESSING VIDEO ---
def process_video(video_path):
    """
    Membaca video dari path sementara, mengekstrak wajah, dan resize.
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(video_path)
    
    frames = []
    
    # Progress bar di UI
    progress_text = "Memproses frame video..."
    my_bar = st.progress(0, text=progress_text)
    
    while len(frames) < SEQUENCE_LENGTH:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Ambil wajah terbesar
            (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
            face = frame[y:y+h, x:x+w]
            resized_face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            
            # NORMALISASI (PENTING: Sesuaikan dengan training Anda)
            # resized_face = resized_face / 255.0  # Uncomment jika training pakai normalisasi
            
            frames.append(resized_face)
            
            # Update progress bar
            percent_complete = int((len(frames) / SEQUENCE_LENGTH) * 100)
            my_bar.progress(percent_complete, text=f"Mengekstrak wajah: {len(frames)}/{SEQUENCE_LENGTH}")

    cap.release()
    my_bar.empty() # Hapus progress bar setelah selesai

    if len(frames) == SEQUENCE_LENGTH:
        return np.expand_dims(np.array(frames), axis=0)
    else:
        return None

# --- MAIN UI ---
def main():
    st.title("🕵️ Deepfake Detection System")
    st.write("Unggah video wajah untuk mendeteksi apakah **REAL** atau **FAKE**.")
    
    # 1. Load Model
    model = load_trained_model()
    
    if model is None:
        return # Stop jika model tidak ada

    # 2. File Uploader
    uploaded_file = st.file_uploader("Pilih file video (.mp4)", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        # Simpan file yang diupload ke tempfile agar bisa dibaca cv2
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        # Tampilkan Video Player
        st.video(tfile.name)
        
        # Tombol Prediksi
        if st.button("🔍 Analisis Video"):
            with st.spinner('Sedang menganalisis kecerdasan buatan...'):
                
                # Proses Video
                input_data = process_video(tfile.name)
                
                if input_data is not None:
                    # Prediksi
                    prediction = model.predict(input_data)
                    confidence = prediction[0][0] # Asumsi output sigmoid (0-1)
                    
                    # Logika Label (Sesuaikan dengan Label Encoder Anda: 0=Real, 1=Fake atau sebaliknya)
                    # Misal: 0 = Fake, 1 = Real
                    # Jika confidence > 0.5 maka Fake (1), else Real (0)
                    
                    is_real = confidence > 0.5
                    final_label = "FAKE" if is_real else "REAL"
                    final_prob = confidence if is_real else (1 - confidence)
                    
                    # Tampilkan Hasil
                    st.divider()
                    if final_label == "FAKE":
                        st.error(f"### HASIL: {final_label} DETECTED! 🚨")
                        st.write("Video ini kemungkinan besar hasil manipulasi AI.")
                    else:
                        st.success(f"### HASIL: {final_label} VIDEO ✅")
                        st.write("Video ini terdeteksi sebagai video asli.")
                    
                    # Tampilkan Meteran Confidence
                    st.write(f"**Tingkat Keyakinan (Confidence): {final_prob*100:.2f}%**")
                    st.progress(int(final_prob * 100))
                    
                else:
                    st.warning(f"Gagal mendeteksi wajah yang cukup. Diperlukan minimal {SEQUENCE_LENGTH} frame wajah yang jelas.")
        
        # Bersihkan file temp
        # os.remove(tfile.name) # Opsional: Hapus file temp setelah selesai

if __name__ == "__main__":
    main()