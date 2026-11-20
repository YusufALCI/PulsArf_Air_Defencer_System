"""
Çelik Kubbe - YOLO ile balon tespiti + orta nokta + PID ile pan/tilt takip

Çalıştırma:
    python balon_takip_pid.py                      # Open Images V7 modeli (Balloon sınıfı hazır)
    python balon_takip_pid.py --model best.pt      # Kendi eğittiğiniz / Roboflow'dan aldığınız model
    python balon_takip_pid.py --port COM3          # Arduino'ya komut göndermek için (Linux: /dev/ttyUSB0)

Akış:  kamera karesi -> YOLO + ByteTrack -> kırmızı (hedef) / mavi (dost) ayrımı
       -> kilitli kırmızı balon (ID) -> orta nokta (cx, cy) + iz + hız
       -> hata = orta nokta - ekran merkezi -> PID (pan ve tilt ayrı) -> Arduino '<dy,dx>'
Mavi balonlar ekranda DOST olarak gösterilir, kilitlenilmez.
"""

import argparse
import time

import cv2
from ultralytics import YOLO

try:
    import serial  # pyserial
except ImportError:
    serial = None



class PID:
    """Basit, motor takibi için yeterli PID.
    - Çıkış sınırlama (motor/servo limitleri için)
    - Integral anti-windup (hedef kaybolunca integral şişmesin)
    - Ölü bölge (merkeze çok yakınken titremesin)
    """

    def __init__(self, kp, ki, kd, out_limit=10.0, i_limit=50.0, deadband=0.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_limit = out_limit
        self.i_limit = i_limit
        self.deadband = deadband
        self.reset()

    def reset(self):
        self._integral = 0.0
        self._prev_err = None
        self._prev_t = None

    def update(self, error):
        now = time.monotonic()
        dt = (now - self._prev_t) if self._prev_t else 0.0
        self._prev_t = now

        if abs(error) < self.deadband:
            # Merkeze yeterince yakın: motoru oynatma. Hatayı yine kaydet ki
            # ölü bölgeden çıkınca türev terimi ani sıçrama yapmasın.
            self._prev_err = error
            return 0.0

        p = self.kp * error

        if dt > 0:
            self._integral += error * dt
            self._integral = max(-self.i_limit, min(self.i_limit, self._integral))
        i = self.ki * self._integral

        d = 0.0
        if self._prev_err is not None and dt > 1e-3:   # çok küçük dt türevi patlatır
            d = self.kd * (error - self._prev_err) / dt
        self._prev_err = error

        out = p + i + d
        return max(-self.out_limit, min(self.out_limit, out))



# ----------------------------------------------------------------------------
# Yardımcılar
# ----------------------------------------------------------------------------
def sinif_gruplari(model):
    """Modeldeki sınıfları hedef (düşman) ve dost olarak ayırır.
    - Kendi modelimiz (red_balloon / blue_balloon): kırmızı = hedef, mavi = dost.
    - Open Images V7 ('Balloon'): renk ayrımı yok, her balon hedef, dost yok.
    Dönüş: (hedef_idleri, dost_idleri)"""
    def balon_mu(ad):
        return "balloon" in ad or "balon" in ad

    adlar = {i: ad.lower() for i, ad in model.names.items()}
    hedef = [i for i, ad in adlar.items() if balon_mu(ad) and ("red" in ad or "kirmizi" in ad)]
    dost = [i for i, ad in adlar.items() if balon_mu(ad) and ("blue" in ad or "mavi" in ad)]
    if not hedef:
        hedef = [i for i, ad in adlar.items() if balon_mu(ad)]
    return hedef, dost


def tespitleri_ayir(result, hedef_siniflar):
    """YOLO sonucunu hedef ve dost listelerine ayırır.
    Her tespit: dict(id, cx, cy, kutu, conf). YOLO kutuyu zaten
    xywh (orta_x, orta_y, genişlik, yükseklik) olarak verir — orta nokta hazır."""
    hedefler, dostlar = [], []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return hedefler, dostlar
    ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(boxes)
    for k in range(len(boxes)):
        cx, cy, _, _ = boxes.xywh[k].tolist()
        t = dict(id=ids[k], cx=cx, cy=cy, kutu=tuple(boxes.xyxy[k].int().tolist()),
                 conf=float(boxes.conf[k]))
        (hedefler if int(boxes.cls[k]) in hedef_siniflar else dostlar).append(t)
    return hedefler, dostlar


def hedef_sec(hedefler, hedef_id):
    """ByteTrack ID'sine göre kilitlenme.
    - Kilitli hedef hâlâ görünüyorsa onu döndürür (başka balon girse de hedef değişmez).
    - Yoksa en yüksek güvenli hedefe kilitlenir.
    Dönüş: (tespit veya None, hedef_id)"""
    if not hedefler:
        return None, hedef_id
    for t in hedefler:
        if hedef_id is not None and t["id"] == hedef_id:
            return t, hedef_id
    t = max(hedefler, key=lambda t: t["conf"])
    return t, t["id"]


def dost_hatta_mi(dostlar, merkez, pay=10):
    """Nişangah bir dost (mavi) balonun kutusunun içinde ya da çok yakınındaysa True.
    Atış eklendiğinde bu durumda ateş edilmeyecek."""
    mx, my = merkez
    return any(x1 - pay <= mx <= x2 + pay and y1 - pay <= my <= y2 + pay
               for x1, y1, x2, y2 in (d["kutu"] for d in dostlar))


class MerkezTakip:
    """Kilitli hedefin orta noktasını kareler boyunca izler.
    - iz: son N orta nokta (ekranda kuyruk olarak çizilir)
    - hız: piksel/saniye (son iki ölçümden, yumuşatılmış)
    - tahmin: hedefin 'onde' saniye sonra nerede olacağı (sabit hız varsayımı).
      Kamera -> model -> motor zincirindeki gecikmeyi telafi etmek için kullanılır."""

    def __init__(self, uzunluk=30, yumusatma=0.5):
        self.uzunluk = uzunluk
        self.a = yumusatma
        self.sifirla()

    def sifirla(self):
        self.iz = []
        self.vx = self.vy = 0.0
        self._son = None

    def guncelle(self, cx, cy):
        now = time.monotonic()
        if self._son is not None:
            px, py, pt = self._son
            dt = now - pt
            if dt > 1e-3:
                # Üstel yumuşatma: tek karelik tespit sıçramaları hızı bozmasın
                self.vx = self.a * self.vx + (1 - self.a) * (cx - px) / dt
                self.vy = self.a * self.vy + (1 - self.a) * (cy - py) / dt
        self._son = (cx, cy, now)
        self.iz.append((int(cx), int(cy)))
        self.iz = self.iz[-self.uzunluk:]

    def tahmin(self, onde):
        cx, cy, _ = self._son
        return cx + self.vx * onde, cy + self.vy * onde


def motor_komutu_gonder(ser, dx, dy):
    """Takımın Arduino koduna (Pulsarf_ciftmotor.ino) göre '<dy,dx>' gönderir.
    dx, dy: motorun şu anki konumuna göre GÖRELİ adım sayısı (tamsayı).
    Sıra önemli: önce dy (Y motoru), sonra dx (X motoru)."""
    if ser is None:
        return
    ser.write(f"<{int(dy)},{int(dx)}>".encode())


def kutu_ciz(frame, t, renk, yazi, kalin=2):
    x1, y1, x2, y2 = t["kutu"]
    cv2.rectangle(frame, (x1, y1), (x2, y2), renk, kalin)
    cv2.putText(frame, yazi, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, renk, 2)


# ----------------------------------------------------------------------------
# Ana döngü
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n-oiv7.pt",
                    help="yolov8n-oiv7.pt (hazır, Balloon sınıfı var) veya kendi best.pt'niz")
    ap.add_argument("--kamera", default=0, help="kamera index'i veya video dosyası")
    ap.add_argument("--port", default=None, help="Arduino seri portu, örn. COM9 / /dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--takipci", default="bytetrack.yaml",
                    help="bytetrack.yaml veya botsort.yaml; 'yok' yazarsanız takip kapalı")
    ap.add_argument("--ayna", action="store_true",
                    help="görüntüyü yatay aynala (eski PulsarfBaloonTespit.py'deki cv2.flip gibi)")
    ap.add_argument("--onde", type=float, default=0.0,
                    help="saniye: PID'e hedefin bu kadar sonraki tahmini yerini ver (ör. 0.1). 0 = kapalı")
    args = ap.parse_args()
    takip_acik = args.takipci != "yok"

    model = YOLO(args.model)
    hedef_siniflar, dost_siniflar = sinif_gruplari(model)
    print("Hedef sınıflar:", [model.names[i] for i in hedef_siniflar])
    print("Dost sınıflar: ", [model.names[i] for i in dost_siniflar] or "yok (model renk ayırmıyor)")
    siniflar = hedef_siniflar + dost_siniflar

    kaynak = int(args.kamera) if str(args.kamera).isdigit() else args.kamera
    cap = cv2.VideoCapture(kaynak)
    if not cap.isOpened():
        raise SystemExit(f"Kamera açılamadı: {args.kamera}")

    ser = None
    if args.port:
        if serial is None:
            raise SystemExit("pyserial yüklü değil: pip install pyserial")
        ser = serial.Serial(args.port, args.baud, timeout=0)
        time.sleep(2)  # Arduino reset sonrası bekleme

    # Hata piksel, çıkış step motor adımı (tamsayıya yuvarlanır).
    # kp=0.5: eski koddaki GAIN_X/GAIN_Y ile aynı başlangıç. Sahada ayarlayın.
    pid_pan = PID(kp=0.5, ki=0.0, kd=0.0, out_limit=200, deadband=15)
    pid_tilt = PID(kp=0.5, ki=0.0, kd=0.0, out_limit=200, deadband=15)
    iz = MerkezTakip()

    kayip_sayac = 0
    hedef_id = None          # ByteTrack'in verdiği, kilitlendiğimiz balonun ID'si
    fps_t = time.monotonic()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.ayna:
            frame = cv2.flip(frame, 1)
        H, W = frame.shape[:2]
        merkez = (W / 2, H / 2)

        if takip_acik:
            # persist=True: takipçi kareler arasında hafızasını korur (ID'ler sabit kalır)
            result = model.track(frame, persist=True, tracker=args.takipci,
                                 conf=args.conf, classes=siniflar,
                                 imgsz=args.imgsz, verbose=False)[0]
        else:
            result = model.predict(frame, conf=args.conf, classes=siniflar,
                                   imgsz=args.imgsz, verbose=False)[0]
        hedefler, dostlar = tespitleri_ayir(result, hedef_siniflar)
        tespit, hedef_id = hedef_sec(hedefler, hedef_id)

        # Dost balonlar: mavi kutu, takip edilmez
        for d in dostlar:
            kutu_ciz(frame, d, (255, 120, 0), f"DOST {d['conf']:.2f}")
        # Kilitlenilmeyen diğer kırmızılar: ince kutu
        for t in hedefler:
            if t is not tespit:
                kutu_ciz(frame, t, (0, 0, 200), f"hedef {t['conf']:.2f}", kalin=1)

        if tespit:
            if kayip_sayac >= 5:          # yeni kilit: eski izi temizle
                iz.sifirla()
            kayip_sayac = 0
            cx, cy = tespit["cx"], tespit["cy"]
            iz.guncelle(cx, cy)

            # Nişan noktası: şu anki merkez ya da gecikme telafili tahmin
            nx, ny = iz.tahmin(args.onde) if args.onde > 0 else (cx, cy)
            hata_x = nx - merkez[0]   # + ise balon sağda
            hata_y = ny - merkez[1]   # + ise balon aşağıda

            dx = round(pid_pan.update(hata_x))
            # Görüntüde Y aşağı doğru artar; motor yukarı dönsün diye eksi
            dy = round(-pid_tilt.update(hata_y))
            if dx or dy:
                motor_komutu_gonder(ser, dx, dy)

            # Çizimler: kilitli kutu, orta nokta, iz, tahmin, hata çizgisi
            kutu_ciz(frame, tespit, (0, 0, 255), f"KILIT ID {tespit['id']}  {tespit['conf']:.2f}", kalin=3)
            for a, b in zip(iz.iz, iz.iz[1:]):
                cv2.line(frame, a, b, (0, 200, 255), 2)
            cv2.circle(frame, (int(cx), int(cy)), 6, (0, 255, 0), -1)
            if args.onde > 0:
                cv2.circle(frame, (int(nx), int(ny)), 6, (255, 0, 255), 2)
            cv2.line(frame, (int(merkez[0]), int(merkez[1])), (int(nx), int(ny)), (255, 255, 0), 1)

            cv2.putText(frame, f"merkez ({cx:.0f},{cy:.0f})  hata ({hata_x:+.0f},{hata_y:+.0f})",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"hiz ({iz.vx:+.0f},{iz.vy:+.0f}) px/s   komut <{dy},{dx}>",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            kayip_sayac += 1
            if kayip_sayac == 5:          # birkaç kare görmezse dur, PID'i ve kilidi sıfırla
                pid_pan.reset()
                pid_tilt.reset()
                hedef_id = None
                motor_komutu_gonder(ser, 0, 0)   # <0,0>: hedefi şu anki konum yap = dur
            cv2.putText(frame, "HEDEF YOK", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Güvenlik göstergesi: nişangah dost balonun üstündeyse
        if dost_hatta_mi(dostlar, merkez):
            cv2.putText(frame, "DOST HATTA - ATES YOK", (10, H - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 120, 0), 2)

        # nişangah
        cv2.drawMarker(frame, (int(merkez[0]), int(merkez[1])), (255, 255, 255),
                       cv2.MARKER_CROSS, 30, 1)
        now = time.monotonic()
        cv2.putText(frame, f"FPS {1 / max(now - fps_t, 1e-6):.1f}", (W - 110, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"hedef {len(hedefler)}  dost {len(dostlar)}", (W - 170, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        fps_t = now

        cv2.imshow("Celik Kubbe - Balon Takip", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    motor_komutu_gonder(ser, 0, 0)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
# PID AYAR NOTLARI (sahada)
# Çıkış birimi: step motor adımı. Arduino'da MAX_SPEED_X=30, MAX_SPEED_Y=25 adım/sn
# olduğu için motor hızlı tepki veremez; çok yavaşsa önce Arduino'daki MAX_SPEED'i artırın.
# 1) ki=0, kd=0 ile başlayın. kp'yi balon merkeze gelip hafif salınana kadar artırın.
# 2) Salınım varsa kd ekleyin (kp'nin ~1/10'u ile başlayın).
# 3) Hedef merkeze tam oturmuyorsa (sabit kayma) küçük bir ki ekleyin.
# 4) deadband: merkezde titreme varsa artırın (eski kodda 30 px idi).
# 5) Motor ters dönüyorsa: dx veya dy satırındaki işareti çevirin.
# 6) Hareketli balonda nişan geride kalıyorsa --onde 0.1 ile başlayın.
# ----------------------------------------------------------------------------
