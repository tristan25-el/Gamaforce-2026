import cv2
import numpy as np

def nothing(x):
    pass

# Baca gambar (sesuaikan dengan nama file Anda)
img = cv2.imread("/home/tristan/Desktop/Gamaforce/Bagian B_soal 2/input-cv-2.jpg")
# Opsional: resize jika gambar terlalu besar agar muat di layar
img = cv2.resize(img, (800, 600))
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.namedWindow("Trackbars")
# Buat 6 slider untuk batas bawah dan atas HSV
cv2.createTrackbar("L_H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L_S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L_V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U_H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U_S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U_V", "Trackbars", 255, 255, nothing)

while True:
    l_h = cv2.getTrackbarPos("L_H", "Trackbars")
    l_s = cv2.getTrackbarPos("L_S", "Trackbars")
    l_v = cv2.getTrackbarPos("L_V", "Trackbars")
    u_h = cv2.getTrackbarPos("U_H", "Trackbars")
    u_s = cv2.getTrackbarPos("U_S", "Trackbars")
    u_v = cv2.getTrackbarPos("U_V", "Trackbars")

    lower_bound = np.array([l_h, l_s, l_v])
    upper_bound = np.array([u_h, u_s, u_v])

    # Lakukan masking warna
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    result = cv2.bitwise_and(img, img, mask=mask)

    # Tampilkan jendela
    cv2.imshow("Masking (Hitam-Putih)", mask)
    cv2.imshow("Hasil", result)

    # Tekan ESC untuk keluar
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
