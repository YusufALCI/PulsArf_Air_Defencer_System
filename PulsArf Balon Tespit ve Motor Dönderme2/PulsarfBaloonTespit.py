import cv2
import numpy as np
import serial
import time

# ============================================================
# AYARLAR
# ============================================================

SERIAL_PORT = "COM9"   # Arduino portunu değiştir
BAUD_RATE = 115200

# Kamera
CAMERA_INDEX = 1

# Görüntü merkezi etrafında tolerans bölgesi
DEAD_ZONE_X = 30
DEAD_ZONE_Y = 30

# Motor hareket katsayısı
GAIN_X = 0.5
GAIN_Y = 0.5

# Çok küçük baloncuk/gürültüleri filtrelemek için
MIN_AREA = 500


# ============================================================
# ARDUINO BAĞLANTISI
# ============================================================

try:
    arduino = serial.Serial(
        SERIAL_PORT,
        BAUD_RATE,
        timeout=0.1
    )

    time.sleep(2)

    print(f"[OK] Arduino bağlantısı: {SERIAL_PORT}")

except Exception as e:
    print("[HATA] Arduino bağlantısı kurulamadı:")
    print(e)
    exit()


# ============================================================
# KAMERA
# ============================================================

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("[HATA] Kamera açılamadı.")
    arduino.close()
    exit()

print("[OK] Kamera açıldı.")
print("Çıkmak için 'q' tuşuna bas.")


# ============================================================
# ANA DÖNGÜ
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("[HATA] Kameradan görüntü alınamadı.")
        break

    # --------------------------------------------------------
    # Görüntüyü Yatay Olarak Aynala (Ayna Efekti)
    # --------------------------------------------------------
    frame = cv2.flip(frame, 1)

    # --------------------------------------------------------
    # Görüntü boyutu
    # --------------------------------------------------------

    height, width = frame.shape[:2]

    center_x = width // 2
    center_y = height // 2

    # --------------------------------------------------------
    # BGR -> HSV
    # --------------------------------------------------------

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # --------------------------------------------------------
    # KIRMIZI RENK MASKESİ
    #
    # Kırmızı HSV'de iki bölgede bulunabilir:
    # 0 civarı ve 180 civarı
    # --------------------------------------------------------

    lower_red_1 = np.array([0, 120, 70])
    upper_red_1 = np.array([10, 255, 255])

    lower_red_2 = np.array([170, 120, 70])
    upper_red_2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(
        hsv,
        lower_red_1,
        upper_red_1
    )

    mask2 = cv2.inRange(
        hsv,
        lower_red_2,
        upper_red_2
    )

    mask = mask1 + mask2

    # --------------------------------------------------------
    # Gürültü temizleme
    # --------------------------------------------------------

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # --------------------------------------------------------
    # KONTURLARI BUL
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    balloon_found = False

    if contours:

        # En büyük kırmızı nesne
        contour = max(
            contours,
            key=cv2.contourArea
        )

        area = cv2.contourArea(contour)

        if area > MIN_AREA:

            # ------------------------------------------------
            # Balon merkezi
            # ------------------------------------------------

            M = cv2.moments(contour)

            if M["m00"] != 0:

                balloon_x = int(
                    M["m10"] / M["m00"]
                )

                balloon_y = int(
                    M["m01"] / M["m00"]
                )

                balloon_found = True

                # --------------------------------------------
                # Balonu çiz
                # --------------------------------------------

                x, y, w, h = cv2.boundingRect(contour)

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                cv2.circle(
                    frame,
                    (balloon_x, balloon_y),
                    8,
                    (255, 0, 0),
                    -1
                )

                # --------------------------------------------
                # Kamera merkezi -> balon farkı
                # --------------------------------------------

                error_x = balloon_x - center_x
                error_y = balloon_y - center_y

                # --------------------------------------------
                # Ekran merkezini çiz
                # --------------------------------------------

                cv2.circle(
                    frame,
                    (center_x, center_y),
                    8,
                    (0, 255, 255),
                    -1
                )

                cv2.line(
                    frame,
                    (center_x, center_y),
                    (balloon_x, balloon_y),
                    (255, 255, 0),
                    2
                )

                # --------------------------------------------
                # Dead-zone
                # --------------------------------------------

                if abs(error_x) < DEAD_ZONE_X:
                    error_x = 0

                if abs(error_y) < DEAD_ZONE_Y:
                    error_y = 0

                # --------------------------------------------
                # Motor hareketi
                # --------------------------------------------

                dx = int(error_x * GAIN_X)

                # Görüntü koordinatında Y aşağı doğru artar.
                # Motor koordinatında ters yönde olması
                # gerekiyorsa eksi işareti kullanıyoruz.
                dy = int(-error_y * GAIN_Y)

                # --------------------------------------------
                # Arduino'ya gönder
                # --------------------------------------------

                if dx != 0 or dy != 0:

                    command = f"<{dy},{dx}>"

                    arduino.write(
                        command.encode()
                    )

                    print(
                        f"Balon: ({balloon_x}, {balloon_y}) | "
                        f"Hata: ({error_x}, {error_y}) | "
                        f"Komut: {command}"
                    )

                # --------------------------------------------
                # Bilgi
                # --------------------------------------------

                cv2.putText(
                    frame,
                    f"Error X: {error_x}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Error Y: {error_y}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

    # --------------------------------------------------------
    # Balon bulunamadı
    # --------------------------------------------------------

    if not balloon_found:

        cv2.putText(
            frame,
            "KIRMIZI BALON BULUNAMADI",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    # --------------------------------------------------------
    # Görüntü merkezi
    # --------------------------------------------------------

    cv2.line(
        frame,
        (center_x - 20, center_y),
        (center_x + 20, center_y),
        (255, 255, 255),
        1
    )

    cv2.line(
        frame,
        (center_x, center_y - 20),
        (center_x, center_y + 20),
        (255, 255, 255),
        1
    )

    # --------------------------------------------------------
    # Görüntüyü göster
    # --------------------------------------------------------

    cv2.imshow(
        "Kirmizi Balon Takibi",
        frame
    )

    cv2.imshow(
        "Kirmizi Maske",
        mask
    )

    # q -> çıkış
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# ============================================================
# TEMİZLE
# ============================================================

cap.release()
arduino.close()
cv2.destroyAllWindows()

print("Program sonlandırıldı.")