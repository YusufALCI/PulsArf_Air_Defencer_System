# PulsArf Air Defense System 🎯

Bir otonom hava savunma ve hedef angajman sistemi için geliştirilmiş, yapay zeka destekli kontrol yazılımı ve kullanıcı arayüzü. 

## 🚀 Proje Hakkında
PulsArf, PC ve NVIDIA Jetson platformları üzerinde çalışmak üzere tasarlanmış; otonom hedef tespiti, takibi ve imhası (simüle edilmiş) süreçlerini yöneten bir sistemdir. Görüntü işleme (YOLO) ve nesne takibi (DeepSORT) algoritmalarını, Arduino tabanlı bir donanım kontrolcüsü ile asenkron (multi-threaded) bir mimaride birleştirir. Sistem, operatöre sahadaki durumsal farkındalığı sağlarken, esnek operasyon modlarıyla görev kritik anlarda kontrolü otonom zekaya devredebilir.

## ✨ Temel Özellikler

* **Yapay Zeka Destekli Algılama (Vision):** YOLOv8 kullanılarak gerçek zamanlı nesne tespiti, sınıflandırma ve "Dost/Düşman" (IFF) ayrımı.
* **Kesintisiz Hedef Takibi:** DeepSORT algoritması ile hareketli hedeflere benzersiz ID atanması; hedeflerin birbirini perdelemesi (occlusion) durumunda dahi takibin sürdürülmesi.
* **Üç Kademeli Operasyon Modu:**
  * **Manuel Mod:** Arayüz (WASD) veya fiziksel Joystick üzerinden kullanıcının taret/motor sistemine doğrudan müdahalesi.
  * **Otonom Mod:** Sistemin çevreyi taraması ve tehditleri otomatik algılaması.
  * **Angajman Modu:** Tehdit önceliklendirmesi yapılarak hedefe kilitlenilmesi ve hedefin vurulana/kadrajdan çıkana kadar otonom olarak merkezde tutulması.
* **Asenkron Yazılım Mimarisi (Multi-threading):** PyQt5 `QThread` ve Signal/Slot yapısı sayesinde; ağır yapay zeka çıkarımları (inference), arayüz (GUI) güncellemeleri ve Arduino seri haberleşmesi birbirini dondurmadan eşzamanlı çalışır.
* **Gelişmiş Kinematik ve PID:** Kameradan alınan piksel sapmalarını (error_x, error_y), motorların ihtiyaç duyduğu hassas derece/step açılarına çeviren kapalı çevrim kontrol sistemi.

## 📂 Sistem Mimarisi ve Klasör Yapısı

Proje, "Sensör (Göz) -> Karar (Beyin) -> Eylem (Motor)" prensibiyle modüler olarak tasarlanmıştır.

```text
PythonProject [PulsArf_Air_Defencer_System]/
├── .venv/                              # Python sanal ortamı
├── AIModelDatas/                       # Eğitim veri setleri ve etiketler
├── AIModels/                           # YOLOv8 (.pt / .engine) model ağırlıkları
├── ArayuzXMLDosyasıBuraya/             # PyQt5 UI tasarım dosyaları (.ui)
├── ConfigTools/
│   └── Configuration.py                # Sistem ayarları, port bilgileri ve sabitler
├── Connections/
│   └── ArduinoConnection.py            # UART üzerinden seri haberleşme ve paketleme
├── DecisionNode/
│   ├── AngajmanMode.py                 # Hedef önceliklendirme ve kilitlenme algoritmaları
│   ├── ManuelMode.py                   # Kullanıcı girdilerini işleme
│   └── OtonomusMode.py                 # Arama ve takip algoritmaları
├── KinematicAndTargetTrackingPID/      # Piksel hatasını motor adımına çeviren PID denetleyicisi
├── SensorsAndCamera/
│   └── CameraFrame.py                  # [QThread] YOLO+DeepSORT koşturan bağımsız görüntü işleme iş parçacığı
├── UserInterfaceControllers/           # Arayüz event'lerini yakalayan kontrolcüler
│   ├── CameraPageController.py
│   ├── ConnectionAndLogPageController.py
│   └── MainPageController.py
├── main.py                             # Tüm sistemi başlatan, sinyalleri dağıtan Router (Hub)
└── README.md

