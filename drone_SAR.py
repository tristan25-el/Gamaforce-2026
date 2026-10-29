import  cv2
import numpy as np
import matplotlib.pyplot as plt

#Mengambil Gambar dan konversi ke RGB dan HSV
img = cv2.imread("/home/tristan/Desktop/Gamaforce/Bagian B_soal 2/input-cv-2.jpg")
img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

img_result = img.copy()


#NILAI THRESHOLD 

# Sumber Bahaya (Merah)
lower_bahaya = np.array([162, 182, 0])
upper_bahaya = np.array([179, 255, 255])

# Korban (Oranye)
lower_korban = np.array([9, 39, 241])
upper_korban = np.array([36, 255, 255])

# Kotak Landzone (Biru) & Lingkaran Tengah (Kuning)
lower_biru = np.array([79, 170, 110])
upper_biru = np.array([109, 255, 255])

lower_kuning = np.array([16, 0, 196])
upper_kuning = np.array([64, 170, 255])


#MASKING 

mask_bahaya = cv2.inRange(img_HSV, lower_bahaya, upper_bahaya)
mask_korban = cv2.inRange(img_HSV, lower_korban, upper_korban)
mask_biru   = cv2.inRange(img_HSV, lower_biru, upper_biru)
mask_kuning = cv2.inRange(img_HSV, lower_kuning, upper_kuning)

#DETEKSI KONTUR & BOUNDING BOX TIAP OBJEK

#  Deteksi Sumber Bahaya 
contours_bahaya, _ = cv2.findContours(mask_bahaya, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours_bahaya:
    c_bahaya = max(contours_bahaya, key=cv2.contourArea)
    x_b, y_b, w_b, h_b = cv2.boundingRect(c_bahaya)
    cx_bahaya = x_b + w_b // 2
    cy_bahaya = y_b + h_b // 2
    
    # Gambar kotak, tanda +, dan teks
    cv2.rectangle(img_result, (x_b, y_b), (x_b + w_b, y_b + h_b), (0, 255, 0), 2)
    cv2.drawMarker(img_result, (cx_bahaya, cy_bahaya), (0, 0, 0), cv2.MARKER_CROSS, 15, 2)
    cv2.putText(img_result, "SUMBER BAHAYA", (x_b, y_b - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Deteksi Korban 
contours_korban, _ = cv2.findContours(mask_korban, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours_korban:
    c_korban = max(contours_korban, key=cv2.contourArea)
    x_k, y_k, w_k, h_k = cv2.boundingRect(c_korban)
    cx_korban = x_k + w_k // 2
    cy_korban = y_k + h_k // 2
    
    # Gambar kotak, tanda +, dan teks
    cv2.rectangle(img_result, (x_k, y_k), (x_k + w_k, y_k + h_k), (0, 255, 0), 2)
    cv2.drawMarker(img_result, (cx_korban, cy_korban), (0, 0, 0), cv2.MARKER_CROSS, 15, 2)
    cv2.putText(img_result, "KORBAN", (x_k, y_k - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Deteksi Landzone 
contours_kuning, _ = cv2.findContours(mask_kuning, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_biru, _   = cv2.findContours(mask_biru, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours_kuning:
    # titik pusat dari lingkaran kuning
    c_kuning = max(contours_kuning, key=cv2.contourArea)
    x_ku, y_ku, w_ku, h_ku = cv2.boundingRect(c_kuning)
    cx_land = x_ku + w_ku // 2
    cy_land = y_ku + h_ku // 2
    
    # kontur kotak biru yang membungkus titik pusat kuning dan bukan laut)
    luas_total_gambar = img.shape[0] * img.shape[1]
    for c_biru in contours_biru:
        luas_biru = cv2.contourArea(c_biru)
        # Abaikan kontur laut 
        if 1000 < luas_biru < (luas_total_gambar * 0.3):
            # Cek apakah titik pusat kuning berada di dalam kontur biru ini
            if cv2.pointPolygonTest(c_biru, (cx_land, cy_land), False) >= 0:
                x_l, y_l, w_l, h_l = cv2.boundingRect(c_biru)
                cv2.rectangle(img_result, (x_l, y_l), (x_l + w_l, y_l + h_l), (0, 255, 0), 2)
                cv2.drawMarker(img_result, (cx_land, cy_land), (0, 0, 0), cv2.MARKER_CROSS, 15, 2)
                cv2.putText(img_result, "LANDZONE", (x_l, y_l - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                break
    # Mendapatkan radius pas lingkaran kuning
    (x_circle, y_circle), radius = cv2.minEnclosingCircle(c_kuning)
    # Gambar lingkaran ungu 
    cv2.circle(img_result, (int(x_circle), int(y_circle)), int(radius), (128, 0, 128), 2)

#Menghitung Jarak,menggambar garis,dan status kritis
if contours_bahaya and contours_korban:
    
    # Hitung jarak Euclidean menggunakan akar pangkat dua bawaan Python
    jarak = ((cx_korban - cx_bahaya)**2 + (cy_korban - cy_bahaya)**2) ** 0.5
    
    # Tarik garis lurus warna kuning (0, 255, 255) ketebalan 2 dari Bahaya ke Korban
    cv2.line(img_result, (cx_bahaya, cy_bahaya), (cx_korban, cy_korban), (0, 255, 255), 2)
    
    # Tulis teks jarak di tengah-tengah garis
    mid_x = (cx_bahaya + cx_korban) // 2
    mid_y = (cy_bahaya + cy_korban) // 2
    teks_jarak = f"Dist: {int(jarak)} px"
    cv2.putText(img_result, teks_jarak, (mid_x + 10, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Logika Status Kritis dengan batas aman 300 px
    batas_aman = 300
    if jarak < batas_aman:
        teks_status = "Status Korban: CRITICAL (terlalu dekat sumber bahaya)"
        warna_status = (255, 255, 255) # Putih
    else:
        teks_status = "Status Korban: AMAN"
        warna_status = (0, 255, 0)     # Hijau
        
    # Tulis teks status di pojok kiri bawah
    tinggi_img = img_result.shape[0]
    cv2.putText(img_result, teks_status, (20, tinggi_img - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, warna_status, 2)

else:
    # Jika salah satu atau kedua objek tidak ada di gambar
    print("Objek Bahaya atau Korban tidak terdeteksi, lewati perhitungan jarak.")

img_result_RGB = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(img_RGB)
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(img_result_RGB)
plt.title("Result")

plt.tight_layout()
plt.show()