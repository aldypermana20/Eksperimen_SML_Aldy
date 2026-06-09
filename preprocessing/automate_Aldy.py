import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import argparse

# --- Konfigurasi Default ---
IMG_SIZE = 128
# Secara default mencari folder dataset dari lokasi script dijalankan
DEFAULT_DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dataset_raw'))
DEFAULT_OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'processed_dataset'))
LABEL_MAP = {'rock': 0, 'paper': 1, 'scissors': 2}

def find_classes_dir(base_path):
    """Mencari lokasi pasti dari folder kelas (rock, paper, scissors)"""
    for root, dirs, files in os.walk(base_path):
        if 'rock' in dirs and 'paper' in dirs and 'scissors' in dirs:
            return root
    return base_path

def load_dataset(dataset_path):
    print(f"Mencari dataset di: {dataset_path}")
    classes_dir = find_classes_dir(dataset_path)
    print(f"Folder kelas terdeteksi di: {classes_dir}")
    
    image_paths = []
    labels = []
    
    for class_name, label in LABEL_MAP.items():
        class_dir = os.path.join(classes_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Direktori kelas '{class_name}' tidak ditemukan di {class_dir}.")
            continue
            
        count = 0
        for file_name in os.listdir(class_dir):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(class_dir, file_name))
                labels.append(label)
                count += 1
        print(f"  - Ditemukan {count} gambar untuk kelas '{class_name}'.")
                
    return image_paths, labels

def preprocess_images(image_paths):
    print(f"\nMemulai preprocessing untuk {len(image_paths)} gambar...")
    data = []
    
    for idx, img_path in enumerate(image_paths):
        img = cv2.imread(img_path)
        if img is not None:
            # Konversi warna
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Resize
            img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            # Normalisasi
            img_normalized = img_resized.astype('float32') / 255.0
            data.append(img_normalized)
        
        if (idx + 1) % 500 == 0:
            print(f"  > Telah memproses {idx + 1} gambar...")
            
    return np.array(data)

def encode_labels(labels):
    # Encode label terintegrasi di tahap load_dataset (LABEL_MAP)
    return np.array(labels)

def split_dataset(data, labels, test_size=0.2):
    print(f"\nMembagi dataset (Train {100-test_size*100:.0f}%, Test {test_size*100:.0f}%)...")
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

def save_dataset(X_train, X_test, y_train, y_test, output_dir):
    print(f"\nMenyimpan dataset ke: {output_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    print("Dataset berhasil disimpan dalam format .npy!")

def main():
    print("="*40)
    print(" PIPELINE AUTOMATION PREPROCESSING")
    print("="*40)
    
    parser = argparse.ArgumentParser(description="Automasi Preprocessing Rock Paper Scissors")
    parser.add_argument('--dataset', type=str, default=DEFAULT_DATASET_PATH, help="Path ke dataset raw")
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_DIR, help="Path untuk menyimpan hasil")
    args = parser.parse_args()
    
    # 1. Load Data
    image_paths, raw_labels = load_dataset(args.dataset)
    if len(image_paths) == 0:
        print("\n[ERROR] Tidak ada gambar yang ditemukan. Pastikan path dataset benar dan dataset sudah diekstrak.")
        return
        
    # 2. Resize & Normalize
    data = preprocess_images(image_paths)
    
    # 3. Encode Label
    labels = encode_labels(raw_labels)
    
    # 4. Split Dataset
    X_train, X_test, y_train, y_test = split_dataset(data, labels, test_size=0.2)
    
    # 5. Save Processed Dataset
    save_dataset(X_train, X_test, y_train, y_test, args.output)
    
    print("\n" + "="*40)
    print(" RANGKUMAN HASIL PREPROCESSING")
    print("="*40)
    print(f"Shape X_train : {X_train.shape}")
    print(f"Shape X_test  : {X_test.shape}")
    print(f"Shape y_train : {y_train.shape}")
    print(f"Shape y_test  : {y_test.shape}")
    print("PIPELINE SELESAI.")

if __name__ == "__main__":
    main()
