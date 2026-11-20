"""
Çelik Kubbe - Balon tespiti (YOLO veya HSV) + orta nokta + PID ile pan/tilt takip

Çalıştırma:
    python balon_takip_pid.py                                 # YOLO (Open Images V7 'Balloon') + HSV ile renk ayrımı
    python balon_takip_pid.py --model best.pt                 # kendi eğittiğiniz red/blue balon modeli
    python balon_takip_pid.py --yontem hsv                    # YOLO'suz, eski PulsarfBaloonTespit.py yöntemi
    python balon_takip_pid.py --kamera 1                      # Arduino portunu kendisi bulur, görüntü aynalı
    python balon_takip_pid.py --no-ayna                       # aynalamayı kapatmak için (varsayılan açık)
    python balon_takip_pid.py --port COM5                     # portu elle vermek için (sadece 5 de olur)
    python balon_takip_pid.py --port yok                      # Arduino'suz (kuru) deneme
    python balon_takip_pid.py --port COM9 --yazdir            # komutları ve Arduino çıktısını terminale yaz
    python balon_takip_pid.py --port COM9 --ters-x            # pan motoru ters dönüyorsa

Akış:  kamera karesi
       -> tespit: YOLO + ByteTrack (bulamazsa HSV maske yedeği)   ya da   HSV maske + kontur (eski kod)
       -> kırmızı (hedef) / mavi (dost) ayrımı: model sınıfından ya da kutu içi HSV renginden
       -> kilitli kırmızı balon (ID / en yakın) -> orta nokta (cx, cy) + iz + hız
       -> hata = orta nokta - ekran merkezi -> PID (pan ve tilt ayrı)
       -> MotorSurucu -> Arduino '<dy,dx>'  (Pulsarf_ciftmotor_claude.ino)
Mavi balonlar ekranda DOST olarak gösterilir, kilitlenilmez.
"""

import argparse
import os
import time

import cv2
import numpy as np

try:
    import serial  # pyserial
except ImportError:
    serial = None

# YOLO (ultralytics) main() içinde, sadece --yontem yolo seçilince yüklenir.
# Böylece --yontem hsv ultralytics kurulu olmayan bilgisayarda da çalışır.


# ============================================================
# AYARLAR (eski PulsarfBaloonTespit.py'den)
# ============================================================
# Kırmızı HSV'de iki uçta bulunur (0 civarı ve 180 civarı), bu yüzden iki aralık.
KIRMIZI_ARALIKLAR = [((0, 120, 70), (10, 255, 255)),
                     ((170, 120, 70), (180, 255, 255))]
MAVI_ARALIKLAR = [((100, 120, 70), (130, 255, 255))]

MIN_ALAN = 500      # HSV yönteminde bundan küçük lekeler gürültü sayılır (piksel²)
RENK_ORANI = 0.15   # YOLO kutusunun iç kısmının en az bu kadarı kırmızı/maviyse o renk sayılır
KAYIP_KARE = 5      # hedef bu kadar kare görünmezse motor durur, kilit ve PID sıfırlanır
CEKIRDEK = np.ones((5, 5), np.uint8)


# ============================================================
# PID
# ============================================================
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


# ============================================================
# MOTOR KONTROL (Arduino seri haberleşme)
# ============================================================
# Arduino ve klon kartlardaki USB-seri çiplerin adları / USB kimlikleri (CH340 = 1A86)
ARDUINO_IZLERI = ("arduino", "ch340", "ch341", "usb-serial", "usb serial", "cp210", "ftdi",
                  "vid:pid=2341:", "vid:pid=1a86:", "vid:pid=10c4:", "vid:pid=0403:")


def portlari_listele():
    """Bağlı seri portlar, hata mesajlarında göstermek için: 'COM5 (USB-SERIAL CH340 (COM5))'"""
    if serial is None:
        return "?"
    from serial.tools import list_ports
    portlar = [f"{p.device} ({p.description})" for p in list_ports.comports()]
    return ", ".join(portlar) or "hiç yok (USB kablosunu kontrol edin)"


def arduino_portu_bul():
    """Arduino gibi görünen ilk seri portu döndürür (CH340 klon kartlar dahil), yoksa None."""
    from serial.tools import list_ports
    for p in list_ports.comports():
        iz = f"{p.description} {p.hwid}".lower()
        if any(k in iz for k in ARDUINO_IZLERI):
            return p.device
    return None


class MotorSurucu:
    """Arduino'daki iki step motora '<dy,dx>' komutu gönderir (Pulsarf_ciftmotor_claude.ino).
    dx, dy: motorun ŞU ANKİ konumuna göre göreli adım sayısı. '<0,0>' = olduğun yerde dur.
    DİKKAT: Orijinal Pulsarf_ciftmotor.ino '<0,0>'ı yok sayar (MultiStepper), motor durmaz.
    Arduino'ya Pulsarf_ciftmotor_claude.ino yüklenmeli.

    - port verilmezse kuru çalışma: komut gönderilmez, kod sadece kamerayla denenebilir.
    - Hedef ölü bölgeye girince BİR KEZ '<0,0>' gönderir. Göndermezse motor, en son verilen
      göreli hedefe kadar gitmeye devam eder ve balonu geçer. Sıfırı tekrar tekrar göndermez.
    - ters_x / ters_y: motor ters dönüyorsa kodda işaret aramadan çevirmek için.
    - Arduino'nun Serial.print ile yazdıklarını okur (yazdir=True ise terminale basar).
    - port 'loop://' gibi bir pyserial URL'si de olabilir (Arduino'suz deneme için).
    """

    def __init__(self, port=None, baud=115200, ters_x=False, ters_y=False, yazdir=False,
                 xy_degistir=False):
        self.ser = None
        self.xy_degistir = xy_degistir   # True: ekrandaki yatay hata Y motoruna, dikey hata X motoruna gider
        self.isaret_x = -1 if ters_x else 1
        self.isaret_y = -1 if ters_y else 1
        self.yazdir = yazdir
        self.son_komut = None      # (dy, dx), en son gönderilen
        self.son_metin = "-"
        self._tampon = ""

        if port is None or str(port).lower() in ("", "yok", "none"):
            print("[UYARI] Port kapalı: kuru çalışma, Arduino'ya komut gönderilmeyecek.")
            return
        if serial is None:
            raise SystemExit("[HATA] pyserial yüklü değil: pip install pyserial")
        port = str(port)
        if port.lower() == "oto":
            port = arduino_portu_bul()
            if port is None:
                print(f"[UYARI] Arduino bulunamadı (bağlı portlar: {portlari_listele()}). "
                      "Kuru çalışma, komut gönderilmeyecek.")
                return
            print(f"[OK] Arduino portu bulundu: {port}")
        elif port.isdigit() and os.name == "nt":
            port = "COM" + port            # '5' -> 'COM5' (Windows'ta port adı böyle)
        try:
            self.ser = serial.serial_for_url(port, baudrate=baud, timeout=0, write_timeout=0.1)
        except (serial.SerialException, ValueError) as e:
            raise SystemExit(
                f"[HATA] Arduino bağlantısı kurulamadı ({port}): {e}\n"
                f"       Bağlı portlar: {portlari_listele()}\n"
                "       Port doğruysa başka bir program kullanıyor olabilir "
                "(Arduino IDE'nin Seri Port Ekranı açık mı?).")
        time.sleep(2)              # Arduino seri port açılınca reset atar; açılmasını bekle
        self.ser.reset_input_buffer()
        print(f"[OK] Arduino bağlantısı: {port} @ {baud}")

    def gonder(self, dx, dy):
        """Komutu gönderir. Gönderdiyse komut metnini, göndermediyse None döndürür."""
        if self.xy_degistir:
            dx, dy = dy, dx        # eksen değişimi, ters çevirmeden (ters_x/ters_y) ÖNCE yapılır
        komut = (int(dy) * self.isaret_y, int(dx) * self.isaret_x)   # sıra: önce Y, sonra X
        if komut == (0, 0) and self.son_komut == (0, 0):
            return None            # zaten durdu, seri hattı boşuna meşgul etme
        metin = f"<{komut[0]},{komut[1]}>"
        if self.ser is not None:
            try:
                self.ser.write(metin.encode())
            except serial.SerialTimeoutException:
                return None        # Arduino tamponu dolu: bu kareyi atla, sonraki kare yine gönderir
        self.son_komut = komut
        self.son_metin = metin
        return metin

    def dur(self):
        return self.gonder(0, 0)

    def oku(self):
        """Arduino'dan gelen satırları okur (okunmazsa PC tarafındaki giriş tamponu dolar)."""
        if self.ser is None or not self.ser.in_waiting:
            return
        self._tampon += self.ser.read(self.ser.in_waiting).decode(errors="ignore")
        *satirlar, self._tampon = self._tampon.split("\n")
        if self.yazdir:
            for s in satirlar:
                if s.strip():
                    print("[Arduino]", s.strip())

    def kapat(self):
        """Programdan çıkarken (hata olsa bile) motoru durdurup portu kapatır."""
        if self.ser is None:
            return
        self.son_komut = None      # '<0,0>'ı kesin göndermek için
        try:
            self.dur()
            self.ser.flush()
        except serial.SerialException as e:   # USB kablosu çıktıysa vb.
            print(f"[UYARI] Durdurma komutu gönderilemedi: {e}")
        finally:
            self.ser.close()


# ============================================================
# HSV renk yardımcıları (eski koddaki maske mantığı)
# ============================================================
def renk_maskesi(hsv, araliklar, temizle=True):
    """Verilen HSV aralıklarının birleşimi olan ikili maske.
    temizle=True: eski koddaki gibi OPEN + CLOSE ile gürültü temizliği."""
    mask = None
    for alt, ust in araliklar:
        m = cv2.inRange(hsv, np.array(alt, np.uint8), np.array(ust, np.uint8))
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    if temizle:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, CEKIRDEK)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, CEKIRDEK)
    return mask


def hsv_tespit(mask, min_alan=MIN_ALAN):
    """Maskedeki her lekeyi tespite çevirir (eski kod sadece en büyüğünü alıyordu;
    burada hepsi listelenir, seçimi hedef_sec yapar). Orta nokta momentlerden.
    Tespit formatı YOLO ile aynı: dict(id, cx, cy, kutu, conf, alan)."""
    tespitler = []
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    for c in contours:
        alan = cv2.contourArea(c)
        if alan < min_alan:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        x, y, w, h = cv2.boundingRect(c)
        tespitler.append(dict(id=None, cx=M["m10"] / M["m00"], cy=M["m01"] / M["m00"],
                              kutu=(x, y, x + w, y + h), conf=1.0, alan=alan, kaynak="HSV"))
    return tespitler


def kutu_rengi(hsv, kutu, ic_pay=0.2, esik=RENK_ORANI):
    """YOLO kutusunun içindeki baskın renk: 'kirmizi', 'mavi' veya None.
    Kutunun kenarlarından %20 içeri alınır: balon yuvarlak, kutu dikdörtgen;
    köşelerdeki arka plan (ör. mavi gökyüzü) renk kararını bozmasın."""
    x1, y1, x2, y2 = (int(v) for v in kutu)
    px, py = int((x2 - x1) * ic_pay), int((y2 - y1) * ic_pay)
    H, W = hsv.shape[:2]
    x1, y1 = max(x1 + px, 0), max(y1 + py, 0)
    x2, y2 = min(x2 - px, W), min(y2 - py, H)
    if x2 <= x1 or y2 <= y1:
        return None
    parca = hsv[y1:y2, x1:x2]
    n = parca.shape[0] * parca.shape[1]
    kirmizi = cv2.countNonZero(renk_maskesi(parca, KIRMIZI_ARALIKLAR, temizle=False)) / n
    mavi = cv2.countNonZero(renk_maskesi(parca, MAVI_ARALIKLAR, temizle=False)) / n
    if max(kirmizi, mavi) < esik:
        return None
    return "kirmizi" if kirmizi >= mavi else "mavi"


# ============================================================
# YOLO yardımcıları
# ============================================================
def sinif_gruplari(model):
    """Modeldeki sınıfları hedef (düşman) ve dost olarak ayırır.
    - Kendi modelimiz (red_balloon / blue_balloon): kırmızı = hedef, mavi = dost.
    - Open Images V7 ('Balloon'): renk ayrımı yok, her balon hedef, dost yok
      (bu durumda main() rengi HSV ile ayırır).
    Dönüş: (hedef_idleri, dost_idleri)"""
    def balon_mu(ad):
        return "balloon" in ad or "balon" in ad

    adlar = {i: ad.lower() for i, ad in model.names.items()}
    hedef = [i for i, ad in adlar.items()
             if balon_mu(ad) and ("red" in ad or "kirmizi" in ad or "kırmızı" in ad)]
    dost = [i for i, ad in adlar.items() if balon_mu(ad) and ("blue" in ad or "mavi" in ad)]
    if not hedef:
        hedef = [i for i, ad in adlar.items() if balon_mu(ad) and i not in dost]
    return hedef, dost


def tespitleri_ayir(result, hedef_siniflar, hsv=None):
    """YOLO sonucunu hedef / dost / belirsiz listelerine ayırır.
    - hsv None: ayrım modelin sınıfına göre (red_balloon -> hedef, diğerleri -> dost).
    - hsv verilirse: ayrım kutu içindeki renge göre (renk bilmeyen modeller için).
      Kırmızı -> hedef, mavi -> dost, ikisi de değilse -> belirsiz (kilitlenilmez).
    Her tespit: dict(id, cx, cy, kutu, conf, alan). YOLO kutuyu zaten
    xywh (orta_x, orta_y, genişlik, yükseklik) olarak verir — orta nokta hazır."""
    hedefler, dostlar, belirsiz = [], [], []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return hedefler, dostlar, belirsiz
    # Tensörleri bir kerede listeye çevir (her kutu için ayrı ayrı CPU'ya taşımaktan hızlı)
    ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(boxes)
    xywh = boxes.xywh.tolist()
    xyxy = boxes.xyxy.int().tolist()
    confs = boxes.conf.tolist()
    clss = boxes.cls.int().tolist()
    for k in range(len(boxes)):
        cx, cy, w, h = xywh[k]
        t = dict(id=ids[k], cx=cx, cy=cy, kutu=tuple(xyxy[k]), conf=float(confs[k]), alan=w * h,
                 kaynak="YOLO")
        if hsv is not None:
            renk = kutu_rengi(hsv, t["kutu"])
            hedef_mi, dost_mu = renk == "kirmizi", renk == "mavi"
        else:
            hedef_mi = clss[k] in hedef_siniflar
            dost_mu = not hedef_mi
        (hedefler if hedef_mi else dostlar if dost_mu else belirsiz).append(t)
    return hedefler, dostlar, belirsiz


# ============================================================
# Takip yardımcıları
# ============================================================
def hedef_sec(hedefler, hedef_id, son_nokta=None, max_sicrama=150):
    """Kilitlenme (başka balon girse de hedef değiştirmeme), sırasıyla:
    1) ByteTrack ID'si kilitli hedefle aynı olan tespit varsa o.
    2) Yoksa bir önceki orta noktaya en yakın hedef (max_sicrama pikselden yakınsa).
       ID'nin olmadığı HSV yönteminde ve ID'nin bir kare kaybolduğu durumda kilidi korur.
    3) Yoksa en yüksek güvenli (HSV'de en büyük alanlı) hedefe yeni kilit.
    Dönüş: (tespit veya None, hedef_id)"""
    if not hedefler:
        return None, hedef_id
    if hedef_id is not None:
        for t in hedefler:
            if t["id"] == hedef_id:
                return t, hedef_id
    if son_nokta is not None:
        sx, sy = son_nokta

        def uzaklik2(t):
            return (t["cx"] - sx) ** 2 + (t["cy"] - sy) ** 2

        t = min(hedefler, key=uzaklik2)
        if uzaklik2(t) <= max_sicrama ** 2:
            return t, t["id"]
    t = max(hedefler, key=lambda t: (t["conf"], t["alan"]))
    return t, t["id"]


def iou(a, b):
    """İki kutunun (x1,y1,x2,y2) kesişim / birleşim oranı."""
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    i = iw * ih
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def hedefleri_dogrula(hedefler, dostlar, belirsiz, hsv):
    """Model 'kırmızı' dediği halde şüpheli olanları hedef listesinden çıkarıp belirsiz yapar
    (kilitlenilmez). Ekip modeli mavi balonu bazen kırmızı sanıyor; yanlışlıkla dost balona
    nişan almak, bir hedefi kaçırmaktan daha kötü olduğu için şüphede hedef alınmaz:
      1) Kutunun içi baskın mavi ise,
      2) Aynı yerde (IoU > 0.4) bir mavi balon tespiti varsa."""
    kalan = []
    for t in hedefler:
        mavi_icinde = kutu_rengi(hsv, t["kutu"]) == "mavi"
        mavi_ustte = any(iou(t["kutu"], d["kutu"]) > 0.4 for d in dostlar)
        (belirsiz if (mavi_icinde or mavi_ustte) else kalan).append(t)
    return kalan, belirsiz


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

    def son_nokta(self):
        return None if self._son is None else self._son[:2]

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


def model_yolu(ad):
    """Sadece dosya adı verildiyse (yolov8n-oiv7.pt) ve terminalin klasöründe yoksa, modeli bu
    .py dosyasının klasöründe arar (yoksa oraya indirir). Aksi halde ultralytics modeli terminal
    hangi klasördeyse (ör. VS Code'un kurulum klasörü) oraya indirir."""
    if os.path.dirname(ad) or os.path.exists(ad):
        return ad
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), ad)


def kutu_ciz(frame, t, renk, yazi, kalin=2):
    x1, y1, x2, y2 = t["kutu"]
    cv2.rectangle(frame, (x1, y1), (x2, y2), renk, kalin)
    cv2.putText(frame, yazi, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, renk, 2)


# ============================================================
# Ana döngü
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="Çelik Kubbe - balon takip (YOLO/HSV + PID + Arduino)")
    ap.add_argument("--yontem", choices=["yolo", "hsv"], default="yolo",
                    help="yolo: YOLO + ByteTrack (varsayılan). hsv: eski renk maskesi yöntemi, YOLO gerekmez")
    ap.add_argument("--model", default="balon_kirmizi_mavi_yolov8n.pt",
                    help="takımın eğittiği kırmızı/mavi balon modeli (varsayılan). "
                         "Hazır alternatif: yolov8n-oiv7.pt (sadece 'Balloon', renk ayırmaz)")
    ap.add_argument("--renk", choices=["oto", "hsv", "model"], default="oto",
                    help="kırmızı/mavi ayrımı. oto: model renk ayırıyorsa modelden, ayırmıyorsa HSV'den. "
                         "hsv: her zaman kutu içi renge bak. model: sadece model sınıfı "
                         "(Open Images'ta her balon hedef olur)")
    ap.add_argument("--kamera", default=0, help="kamera index'i veya video dosyası (eski kodda 1)")
    ap.add_argument("--port", default="oto",
                    help="Arduino seri portu: COM5 (ya da sadece 5) / /dev/ttyUSB0. "
                         "oto = Arduino'yu kendisi bulur (varsayılan), yok = Arduino'suz kuru çalışma")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--conf", type=float, default=None,
                    help="YOLO en düşük güven. Varsayılan: renk ayıran takım modelinde 0.5 "
                         "(mavi balonun kırmızı sanılmasını azaltır), hazır modelde 0.10 "
                         "(o model balonu en fazla ~0.14 güvenle görüyor)")
    ap.add_argument("--yedek-hsv", action=argparse.BooleanOptionalAction, default=None,
                    help="YOLO bulamazsa renk maskesiyle ara. Varsayılan: sadece renk ayırmayan hazır "
                         "modelde açık; takım modelinde kapalı (kırmızı tişört gibi şeylere kilitlenmesin)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--takipci", default="bytetrack.yaml",
                    help="bytetrack.yaml veya botsort.yaml; 'yok' yazarsanız takip kapalı")
    ap.add_argument("--ayna", action=argparse.BooleanOptionalAction, default=True,
                    help="görüntüyü yatay aynala (eski koddaki cv2.flip gibi). "
                         "Varsayılan açık; kapatmak için --no-ayna")
    ap.add_argument("--onde", type=float, default=0.0,
                    help="saniye: PID'e hedefin bu kadar sonraki tahmini yerini ver (ör. 0.1). 0 = kapalı")
    # PID / motor ayarları (sahada kodu değiştirmeden ayarlamak için)
    ap.add_argument("--kp", type=float, default=0.5, help="eski koddaki GAIN_X/GAIN_Y ile aynı başlangıç")
    ap.add_argument("--ki", type=float, default=0.0)
    ap.add_argument("--kd", type=float, default=0.0)
    ap.add_argument("--olu-bolge", type=float, default=15,
                    help="piksel; hata bundan küçükse o eksen oynamaz (eski kodda DEAD_ZONE=30)")
    ap.add_argument("--max-adim", type=float, default=200, help="tek komutta en fazla adım")
    ap.add_argument("--ters-x", action="store_true", help="X motoru ters dönüyorsa (eksen değişiminden sonraki fiziksel motor)")
    ap.add_argument("--ters-y", action="store_true", help="Y motoru ters dönüyorsa (eksen değişiminden sonraki fiziksel motor)")
    ap.add_argument("--xy-degistir", action=argparse.BooleanOptionalAction, default=True,
                    help="X ve Y eksenlerini birbirine çevir: görüntüdeki yatay hata Y motoruna, dikey hata "
                         "X motoruna gider (ek kamera/motor bağlantısı yüzünden eksenler karıştıysa). "
                         "Varsayılan açık; eski düzene dönmek için --no-xy-degistir")
    # HSV yöntemi
    ap.add_argument("--min-alan", type=float, default=MIN_ALAN, help="hsv yönteminde en küçük leke alanı")
    ap.add_argument("--maske", action="store_true",
                    help="renk maskesi penceresini göster (programın kırmızı | mavi gördüğü yerler beyaz)")
    ap.add_argument("--yazdir", action="store_true",
                    help="gönderilen her komutu ve Arduino çıktısını terminale yaz")
    args = ap.parse_args()

    yolo = args.yontem == "yolo"
    takip_acik = yolo and args.takipci != "yok"
    renk_hsv = False

    if yolo:
        from ultralytics import YOLO
        model = YOLO(model_yolu(args.model))
        hedef_siniflar, dost_siniflar = sinif_gruplari(model)
        if not hedef_siniflar:
            raise SystemExit(f"[HATA] Modelde balon sınıfı bulunamadı. Sınıflar: {list(model.names.values())}")
        siniflar = hedef_siniflar + dost_siniflar
        renk_hsv = args.renk == "hsv" or (args.renk == "oto" and not dost_siniflar)
        if args.conf is None:
            args.conf = 0.10 if renk_hsv else 0.5
        if args.yedek_hsv is None:
            args.yedek_hsv = renk_hsv
        print("Yöntem: YOLO", "+ " + args.takipci if takip_acik else "(takip kapalı)")
        print("Balon sınıfları:", [model.names[i] for i in siniflar])
        print(f"YOLO güven eşiği: {args.conf} | HSV yedeği: {'açık' if args.yedek_hsv else 'kapalı'}")
        if renk_hsv:
            print("Renk ayrımı: HSV (kutu içi kırmızı -> hedef, mavi -> dost, diğer -> belirsiz)")
        else:
            print("Hedef sınıflar:", [model.names[i] for i in hedef_siniflar])
            print("Dost sınıflar: ", [model.names[i] for i in dost_siniflar] or "yok")
    else:
        print("Yöntem: HSV renk maskesi (YOLO kapalı). Kırmızı -> hedef, mavi -> dost")

    kaynak = int(1) if str(args.kamera).isdigit() else args.kamera
    cap = cv2.VideoCapture(kaynak)
    if not cap.isOpened() and isinstance(kaynak, int) and os.name == "nt":
        # Windows'ta varsayılan yöntem ikinci/USB kamerayı bazen açamıyor; DirectShow ile dene
        cap.release()
        cap = cv2.VideoCapture(kaynak, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit(f"[HATA] Kamera açılamadı: {args.kamera}. "
                         "Başka program kamerayı kullanıyor olabilir; --kamera 0 / 1 deneyin.")
    print("[OK] Kamera açıldı. Çıkmak için 'q' tuşuna bas.")

    try:
        motor = MotorSurucu(args.port, args.baud, args.ters_x, args.ters_y, args.yazdir,
                            args.xy_degistir)
    except SystemExit:
        cap.release()
        raise

    # Hata piksel, çıkış step motor adımı (tamsayıya yuvarlanır).
    pid_pan = PID(args.kp, args.ki, args.kd, out_limit=args.max_adim, deadband=args.olu_bolge)
    pid_tilt = PID(args.kp, args.ki, args.kd, out_limit=args.max_adim, deadband=args.olu_bolge)
    iz = MerkezTakip()

    def skor(t):
        return f"{t['conf']:.2f}" if t.get("kaynak") == "YOLO" else f"HSV {t['alan']:.0f}px"

    kayip_sayac = 0
    hedef_id = None          # ByteTrack'in verdiği, kilitlendiğimiz balonun ID'si
    fps_t = time.monotonic()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[HATA] Kameradan görüntü alınamadı.")
                break
            if args.ayna:
                frame = cv2.flip(frame, 1)
            H, W = frame.shape[:2]
            merkez = (W / 2, H / 2)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # Renk maskeleri: hsv yönteminde tespitin kendisi; yolo yönteminde yedek ve gösterim için
            if not yolo or args.yedek_hsv or args.maske:
                k_maske = renk_maskesi(hsv, KIRMIZI_ARALIKLAR)
                m_maske = renk_maskesi(hsv, MAVI_ARALIKLAR)
                if args.maske:
                    cv2.imshow("Maske: kirmizi | mavi", np.hstack([k_maske, m_maske]))

            # ---------------- Tespit ----------------
            if yolo:
                if takip_acik:
                    # persist=True: takipçi kareler arasında hafızasını korur (ID'ler sabit kalır)
                    result = model.track(frame, persist=True, tracker=args.takipci,
                                         conf=args.conf, classes=siniflar,
                                         imgsz=args.imgsz, verbose=False)[0]
                else:
                    result = model.predict(frame, conf=args.conf, classes=siniflar,
                                           imgsz=args.imgsz, verbose=False)[0]
                hedefler, dostlar, belirsiz = tespitleri_ayir(result, hedef_siniflar,
                                                               hsv if renk_hsv else None)
                if not renk_hsv:
                    hedefler, belirsiz = hedefleri_dogrula(hedefler, dostlar, belirsiz, hsv)
                if args.yedek_hsv:
                    # Hazır YOLO modeli balonu her karede bulamıyor (kamerada güveni 0.14'ü geçmedi).
                    # Bu karede kırmızı/mavi balon bulamadıysa eski renk maskesi yöntemine düş.
                    if not hedefler:
                        hedefler = hsv_tespit(k_maske, args.min_alan)
                    if not dostlar:
                        dostlar = hsv_tespit(m_maske, args.min_alan)
            else:
                hedefler = hsv_tespit(k_maske, args.min_alan)
                dostlar = hsv_tespit(m_maske, args.min_alan)
                belirsiz = []

            son = iz.son_nokta() if kayip_sayac < KAYIP_KARE else None
            tespit, hedef_id = hedef_sec(hedefler, hedef_id, son)

            # Dost balonlar: mavi kutu, takip edilmez
            for d in dostlar:
                kutu_ciz(frame, d, (255, 120, 0), f"DOST {skor(d)}")
            # Rengi anlaşılmayan balonlar: gri, takip edilmez
            for b in belirsiz:
                kutu_ciz(frame, b, (160, 160, 160), f"renk? {skor(b)}", kalin=1)
            # Kilitlenilmeyen diğer kırmızılar: ince kutu
            for t in hedefler:
                if t is not tespit:
                    kutu_ciz(frame, t, (0, 0, 200), f"hedef {skor(t)}", kalin=1)

            # ---------------- Takip + motor ----------------
            if tespit:
                if kayip_sayac >= KAYIP_KARE:     # yeni kilit: eski izi temizle
                    iz.sifirla()
                kayip_sayac = 0
                cx, cy = tespit["cx"], tespit["cy"]
                iz.guncelle(cx, cy)

                # Nişan noktası: şu anki merkez ya da gecikme telafili tahmin
                nx, ny = iz.tahmin(args.onde) if args.onde > 0 else (cx, cy)
                hata_x = nx - merkez[0]   # + ise balon sağda
                hata_y = ny - merkez[1]   # + ise balon aşağıda

                # Motor yönü aynalamaya bağlı olmasın. Eski PulsarfBaloonTespit.py aynalı görüntüde
                # dx = +hata ile doğru dönüyordu. Aynalama kapalıyken sağ/sol yer değiştirdiği için
                # X hatası ters çevrilir; böylece --ayna / --no-ayna motor yönünü bozmaz.
                hata_x_motor = hata_x if args.ayna else -hata_x
                dx = round(pid_pan.update(hata_x_motor))
                # Görüntüde Y aşağı doğru artar; motor yukarı dönsün diye eksi
                dy = round(-pid_tilt.update(hata_y))
                # İkisi de 0 ise (ölü bölge) MotorSurucu bir kez '<0,0>' gönderip motoru durdurur
                gonderilen = motor.gonder(dx, dy)
                if gonderilen and args.yazdir:
                    print(f"Balon: ({cx:.0f}, {cy:.0f}) | Hata: ({hata_x:+.0f}, {hata_y:+.0f}) "
                          f"| Komut: {gonderilen}")

                # Çizimler: kilitli kutu, orta nokta, iz, tahmin, hata çizgisi
                kimlik = f" ID {tespit['id']}" if tespit["id"] is not None else ""
                kutu_ciz(frame, tespit, (0, 0, 255), f"KILIT{kimlik}  {skor(tespit)}", kalin=3)
                for a, b in zip(iz.iz, iz.iz[1:]):
                    cv2.line(frame, a, b, (0, 200, 255), 2)
                cv2.circle(frame, (int(cx), int(cy)), 6, (0, 255, 0), -1)
                if args.onde > 0:
                    cv2.circle(frame, (int(nx), int(ny)), 6, (255, 0, 255), 2)
                cv2.line(frame, (int(merkez[0]), int(merkez[1])), (int(nx), int(ny)), (255, 255, 0), 1)

                cv2.putText(frame, f"merkez ({cx:.0f},{cy:.0f})  hata ({hata_x:+.0f},{hata_y:+.0f})",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"hiz ({iz.vx:+.0f},{iz.vy:+.0f}) px/s   komut {motor.son_metin}",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                kayip_sayac += 1
                if kayip_sayac == KAYIP_KARE:     # birkaç kare görmezse dur, PID'i ve kilidi sıfırla
                    pid_pan.reset()
                    pid_tilt.reset()
                    hedef_id = None
                    if motor.dur() and args.yazdir:   # <0,0>: hedefi şu anki konum yap = dur
                        print("Hedef kayboldu | Komut: <0,0>")
                cv2.putText(frame, "HEDEF YOK", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            motor.oku()

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
            cv2.putText(frame, "ARDUINO" if motor.ser else "KURU (port yok)", (W - 170, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if motor.ser else (0, 165, 255), 1)
            fps_t = now

            cv2.imshow("Celik Kubbe - Balon Takip", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        # Hata / Ctrl+C olsa bile motor durdurulur ve port kapatılır
        motor.kapat()
        cap.release()
        cv2.destroyAllWindows()
        print("Program sonlandırıldı.")


if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
# PID AYAR NOTLARI (sahada)  — artık kodu değiştirmeden komut satırından ayarlanır
# Çıkış birimi: step motor adımı. Arduino'da MAX_SPEED_X=30, MAX_SPEED_Y=25 adım/sn.
# MultiStepper ivmesiz, sabit MAX_SPEED ile döner. Python her karede (~33 ms) yeni komut
# yolladığı için motor bir karede en fazla MAX_SPEED x 0.033 adım (~1 adım) atabilir:
# komut bundan büyükse motor hep tam hızda döner ve kp hızı DEĞİŞTİRMEZ.
# Önce Arduino'daki MAX_SPEED'i artırın (ör. 600 -> karede ~20 adım); kp ancak o zaman etkili olur.
# 1) --ki 0 --kd 0 ile başlayın. --kp'yi balon merkeze gelip hafif salınana kadar artırın.
# 2) Salınım varsa --kd ekleyin (kp'nin ~1/10'u ile başlayın).
# 3) Hedef merkeze tam oturmuyorsa (sabit kayma) küçük bir --ki ekleyin.
# 4) --olu-bolge: merkezde titreme varsa artırın (eski kodda 30 px idi).
# 5) Motor ters dönüyorsa: --ters-x veya --ters-y. (--ayna / --no-ayna motor yönünü DEĞİŞTİRMEZ;
#    kod bunu kendisi telafi eder. Yön, eski koddaki aynalı görüntüye göre ayarlıdır.)
# 6) Hareketli balonda nişan geride kalıyorsa --onde 0.1 ile başlayın.
# 7) Gökyüzü mavi maskeye giriyorsa (hsv yönteminde sürekli DOST çıkıyorsa) MAVI_ARALIKLAR'daki
#    doygunluk alt sınırını (120) yükseltin veya --yontem yolo kullanın.
# ----------------------------------------------------------------------------
