"""
Kırmızı / mavi balon modeli eğitimi
-----------------------------------
Önce:  python veri_hazirla.py indir/gokhisarmain ...   (veri/balon/data.yaml oluşur)

Çalıştırma:
    python egit.py                         # YOLO11s, 1280 boyut
    python egit.py --model yolo11n.pt      # daha küçük/hızlı model
    python egit.py --imgsz 640 --epochs 30 # hızlı ilk deneme

Sonuç: runs/balon/<ad>/weights/best.pt
       python balon_takip_pid.py --model runs/balon/<ad>/weights/best.pt
"""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="veri/balon/data.yaml")
    ap.add_argument("--model", default="yolo11s.pt", help="yolo11n.pt / yolo11s.pt / yolo26s.pt ...")
    ap.add_argument("--imgsz", type=int, default=1280, help="küçük hedef için 1280; hızlı deneme için 640")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=-1, help="-1: GPU belleğine göre otomatik")
    ap.add_argument("--ad", default=None, help="çıktı klasörünün adı")
    args = ap.parse_args()

    if torch.cuda.is_available():
        cihaz = 0
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        cihaz = "cpu"
        print("UYARI: GPU kullanılamıyor, eğitim CPU'da çok yavaş olur. "
              "CUDA'lı torch kurulumu için NOTLAR.md'ye bakın.")

    model = YOLO(args.model)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch if cihaz != "cpu" else 4,
        device=cihaz,
        # Tam yol: göreli verilirse Ultralytics runs/detect/ altına iç içe yazıyor
        project=str(Path("runs/balon").resolve()),
        name=args.ad or f"{args.model.split('.')[0]}_{args.imgsz}",
        patience=20,       # 20 epoch iyileşme yoksa dur
        # Renk ayrımı sınıf bilgisi: renk tonu kaydırması kırmızıyı maviye çevirebilir, kapalı.
        hsv_h=0.0,
        # Parlaklık / doygunluk değişimi ışık farklarına karşı faydalı, açık kalır.
        hsv_s=0.5,
        hsv_v=0.4,
        fliplr=0.5,
        mosaic=1.0,
        workers=4,         # Windows'ta çok yüksek değer sorun çıkarabiliyor
    )

    metrikler = model.val(data=args.data, imgsz=args.imgsz, split="test" if _test_var(args.data) else "val")
    print(f"\nmAP50: {metrikler.box.map50:.3f}   mAP50-95: {metrikler.box.map:.3f}")
    print("Sınıf bazında mAP50:", {model.names[i]: round(float(v), 3)
                                   for i, v in zip(metrikler.box.ap_class_index, metrikler.box.ap50)})


def _test_var(data_yolu):
    import yaml
    with open(data_yolu, encoding="utf-8") as f:
        return "test" in yaml.safe_load(f)


if __name__ == "__main__":
    main()
