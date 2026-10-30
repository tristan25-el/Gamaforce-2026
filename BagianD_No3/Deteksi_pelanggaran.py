import cv2
from ultralytics import YOLO

def main():
    # 1.model pretrained YOLOv8n menggunakan dataset COCO
    model = YOLO("yolov8n.pt")

    # 2. Baca gambar input motor.jpg
    img = cv2.imread("motor.jpg")
    
    if img is None:
        print("File tidak ditemukan")
        return

    # 3. Deteksi objek khusus kelas motor (COCO class ID 3 = motorcycle)
    results = model(img, classes=[3])

    print("HASIL DETEKSI PELANGGAR K4L")
    
    # 4. Iterasi setiap hasil deteksi
    for result in results:
        for box in result.boxes:
            # Ekstrak koordinat bounding box (x_min, y_min, x_max, y_max)
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
            
            # Ekstrak nilai confidence score
            conf = float(box.conf[0])

            #Cetak koordinat bounding box dan confidence score 
            print(f"Koordinat BBox: (x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}) | Confidence: {conf:.2f}")

            # Gambar bounding box pada objek motor dengan OpenCV
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

            # Ganti teks label menjadi "Pelanggar K4L" dengan OpenCV
            label = f"Pelanggar K4L ({conf:.2f})"
            
            # Buat latar belakang warna merah untuk teks agar mudah dibaca
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x_min, y_min - text_h - 10), (x_min + text_w, y_min), (0, 0, 255), -1)
            
            # Tulis teks label berwarna putih di atas kotak
            cv2.putText(img, label, (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Simpan hasil gambar dengan anotasi menjadi output.jpg
    output_path = "output.jpg"
    cv2.imwrite(output_path, img)
    print(f"\nBerhasil! Gambar hasil anotasi telah disimpan sebagai '{output_path}'.")

if __name__ == "__main__":
    main()