# Çelik Kubbe – Balon Tespit ve Takip Notları

Bu notlar claude.ai'deki "YOLO nesne tespiti planı" sohbetinden aktarıldı.

## Hazır olanlar ve yazılması gerekenler

| Parça | Hazır mı? | Ne kullanıyoruz |
|---|---|---|
| Balon tespit modeli | Hazır | `yolov8n-oiv7.pt` (Open Images V7). İçinde "Balloon" sınıfı var (sınıf no 20). Standart `yolov8n.pt` (COCO) modelinde balon sınıfı yok. |
| Orta nokta | Hazır | `boxes.xywh` → (orta_x, orta_y, genişlik, yükseklik) |
| Takip (ID) | Hazır | Ultralytics içindeki ByteTrack: `model.track(..., tracker="bytetrack.yaml", persist=True)` |
| PID | Hazır kütüphane var | Python'da `simple-pid`, Arduino'da `PID_v1`. Dosyada bağımlılık gerektirmeyen küçük bir PID sınıfı var. |
| Birleştiren kod | Yazıldı | `balon_takip_pid.py` |
| Motor komutu | Donanıma özel | `motor_komutu_gonder` fonksiyonu Arduino koduna göre uyarlanmalı |

Roboflow'daki balon veri setleri: "balloon (Hava Savunma)", "Teknofest 2", "Balloon-Detection" (bu sonuncusu yalnızca 10 görüntü).

## Çalıştırma

```
pip install ultralytics opencv-python pyserial lap
python balon_takip_pid.py                  # sadece görüntü, motor yok
python balon_takip_pid.py --port COM3      # Arduino'ya "P<pan>,T<tilt>" gönderir
python balon_takip_pid.py --model best.pt  # kendi eğittiğiniz model
python balon_takip_pid.py --takipci yok    # takibi kapatır
```

## Bu haftaki görev: adım adım

### Adım 1 – Hazır modelle deneme (eğitim yok)
`yolov8n-oiv7.pt` klasörde, "Balloon" sınıfı (20) doğrulandı.
```
python balon_takip_pid.py
```
Bu model kırmızı ile maviyi ayırmaz, her balona kilitlenir. Sadece zinciri (tespit → orta nokta → PID) denemek için.

### Adım 2 – Veri setini indir
Roboflow Universe'e ücretsiz hesapla giriş yapın → veri setini açın → **Download Dataset** → format **YOLOv8** → **Download zip to computer**. Zip'i `indir/<set_adı>/` içine açın.

| Veri seti | Görüntü | Sınıflar | Not |
|---|---|---|---|
| [gökhisarmain](https://universe.roboflow.com/senadb/gokhisarmain) | 2.334 | balloon, blue balloon, red balloon | **Ana set** |
| [gökhisar3](https://universe.roboflow.com/senadb/gokhisar3) | 420 | blue balloon, red balloon | Aynı yazar, gökhisarmain ile çakışabilir |
| [balloon detect](https://universe.roboflow.com/btrme-hhvgp/balloon-detect-csvm7) | 272 | red, blue, i | Ek veri |

Hepsi CC BY 4.0. Rapor/sunumda kaynak gösterin.

### Adım 3 – Setleri birleştir
```
python veri_hazirla.py indir/gokhisarmain indir/balloon-detect
```
- Sınıflar `red_balloon` (0) ve `blue_balloon` (1) olarak birleşir.
- Renk belirtilmemiş "balloon" kutuları HSV ile kırmızı/mavi ayrılır; belirsizse o görüntü atlanır.
- Aynı görüntü iki sette varsa bir kez alınır.
- Çıktı: `veri/balon/data.yaml`. Özet tablodaki kırmızı/mavi sayılarına bakın.

### Adım 4 – GPU'lu torch (bir kez)
Şu an kurulu torch CPU sürümü (`2.15.0+cpu`), RTX 4060 kullanılmıyor. CPU'da eğitim saatler sürer.
[pytorch.org/get-started](https://pytorch.org/get-started/locally/) sayfasında Windows + Pip + CUDA seçip verdiği komutu çalıştırın, örn.:
```
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.cuda.is_available())"
```
`True` yazmalı.

### Adım 5 – Eğit
```
python egit.py --imgsz 640 --epochs 30    # hızlı ilk deneme
python egit.py                            # asıl eğitim: YOLO11s, 1280
```
Renk tonu augmentasyonu (`hsv_h`) kapalı. Sonuç: `runs/balon/<ad>/weights/best.pt`.

### Adım 6 – Kendi modelinle takip
```
python balon_takip_pid.py --model runs/balon/yolo11s_1280/weights/best.pt
```
Model `red_balloon` sınıfı içerdiğinde kod **yalnızca kırmızıya** kilitlenir, maviyi takip etmez.

### Adım 7 – Motor ve PID
`motor_komutu_gonder`, takım reposundaki `Pulsarf_ciftmotor.ino` (Yusuf dalı) ile uyumlu:
`<dy,dx>` = Y ve X motoruna **göreli** adım sayısı (tamsayı). `<0,0>` = dur.
```
python balon_takip_pid.py --model ../AIModels/balon_kirmizi_mavi_yolo11n.pt --kamera 1 --port COM9
python balon_takip_pid.py ... --ayna        # eski koddaki gibi görüntüyü aynala
python balon_takip_pid.py ... --onde 0.1    # hareketli hedefte 0.1 sn ilerisine nişan al
```
Ekranda: kırmızı = KİLİT (ID, merkez, iz, hız), mavi = DOST (kilitlenilmez),
nişangah mavi balonun üstündeyse "DOST HATTA - ATES YOK".
Arduino'da MAX_SPEED_X=30, MAX_SPEED_Y=25 adım/sn: taret yavaş kalırsa önce bunları artırın, sonra kp ayarı.

## Sistem akışı

Kamera (lens + ayarlar) → ön işleme (ölçekleme, kırpma) → YOLO tespiti → renk doğrulama → takip (ID + tahmin) → karar (ateş / ateş yok) → PID ile nişan → atış

### 1. Lens (AR0234: 1920×1200, 3 µm piksel; balon ~30 cm varsayıldı)

| Lens | Yatay görüş | 15 m | 20 m |
|---|---|---|---|
| 2.8 mm | ~92° | 19 px (640'ta 6) | 14 px (640'ta 5) |
| 4 mm | ~72° | 27 px | 20 px |
| 6 mm | ~51° | 40 px | 30 px |
| 8 mm | ~40° | 53 px | 40 px |
| 12 mm | ~27° | 80 px | 60 px |

- YOLO 10 pikselin altında güvenilmez. Geniş açılı lens yerine **6 mm (ana) + 8 mm (yedek)** önerildi. 12 mm önerilmedi.
- Alırken: M12 bağlantı, görüntü dairesi ≥ 1/2.5", **IR-cut filtre** (renk ayrımı için şart), F2.0–F2.8, düşük bozulma.
- **Netleme ~8 m'ye yapılır.** 6 mm F2.0 ile ~2.2 m → sonsuz, 8 mm ile ~3.2 m → sonsuz net kalır.
- Netleme yöntemi: 8 m'ye çizgili hedef koy, `cv2.Laplacian(gri, cv2.CV_64F).var()` skorunu ekrana yaz, en yüksek skorda lensi kilitle.

### 2. Kamera ayarları
Kısa pozlama (birkaç ms), sabit beyaz dengesi, elle sabitlenmiş odak, tam çözünürlük, USB 3.0 port. Zoom yazılımla yapılır (kırpma).

### 3. Model seçimi
- **YOLO11**: ana öneri.
- **YOLO26**: en yeni, küçük hedefler için iyi; YOLO11 ile yan yana denenmeli.
- YOLOv8: sağlam ama artık ilk tercih değil. YOLOv10: hızlı ama daha az doğru. YOLOv12: gerçek zamanlı kullanım için önerilmedi.
- Boyut n veya s; Jetson'da TensorRT FP16 export.

### 4. Veri ve eğitim
- Sınıflar: `red_balloon`, `blue_balloon`.
- Kendi kamera + lensinizle 5–25 m, farklı ışıklarda, hareketli balon dahil veri çekin.
- Zor negatifler ekleyin (kırmızı tişört, mavi poşet).
- `hsv_h=0` (renk tonu augmentasyonu kapalı).
- `imgsz=1280` ile eğitin; sonuçları mesafeye göre raporlayın.

### 5. Ön işleme
- Arama modu: tam kare, 1280. Takip modu: hedef etrafından 640×640 tam çözünürlüklü kırpma.
- Gerekirse arama modunda SAHI.
- Işık değişkense CLAHE'yi yalnızca Lab'ın L kanalına uygulayın; eğitimde ne yapıldıysa çalışırken de aynısı.

### 6. Dost/düşman ayrımı
1. YOLO sınıfı `red_balloon` ve yüksek güven.
2. HSV renk doğrulama (kırmızı: 0–10 **ve** 170–180 aralıkları).
3. Zamansal oylama: aynı ID, son 10 karenin en az 8'inde kırmızı.

Ateş için üçü de geçmeli ve nişan çizgisi yakınında mavi balon olmamalı. Emin değilsen: **ateş yok**.

### 7. Hareketli hedef
ByteTrack (veya BoT-SORT) ile ID; 50–100 ms gecikmeyi Kalman ile hedefin tahmini yerine nişan alarak telafi edin.

### 8. LiDAR
Tespit için gerekmez. Önce görüntüden mesafe: `mesafe ≈ odak × gerçek çap / kutu genişliği`. Yetmezse namluyla aynı eksende tek noktalı LiDAR.

## PID kısaca
`motor = kp × hata + ki × birikmiş_hata + kd × hatanın_değişimi`

- **P**: hata büyükse hızlı, küçükse yavaş döner.
- **D**: aşırı yaklaşmayı frenler, salınımı söndürür.
- **I**: kalan küçük kaymayı giderir; az kullanılır.

Ayar sırası: `ki=0, kd=0` → kp'yi hafif salınıma kadar artır → kd ekle (~kp/10) → gerekirse çok küçük ki.

## Önerilen sıra
1. Lensi değiştir, kamera ayarlarını manuele al.
2. Kendi verini çek ve etiketle.
3. YOLO11s ve YOLO26s'i 1280'de eğit, mesafeye göre karşılaştır.
4. Kırpma (takip modu) ve renk doğrulamayı ekle.
5. ByteTrack + Kalman + PID'yi bağla.
6. Gerekirse LiDAR ekle.

## Açık sorular
- Modeli hangi bilgisayar çalıştıracak (Jetson / laptop / Raspberry Pi)?
- Şartnamedeki balon çapı?
- Arduino tarafının beklediği komut formatı (`motor_komutu_gonder` buna göre uyarlanacak).
