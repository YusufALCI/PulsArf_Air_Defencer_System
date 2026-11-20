"""
Roboflow'dan indirilen balon veri setlerini tek bir YOLO veri setinde birleştirir.
---------------------------------------------------------------------------------
Sınıflar:  0 = red_balloon   1 = blue_balloon

Kullanım:
    1) Roboflow Universe'te veri setini açın -> "Download Dataset" -> format "YOLOv8"
       -> "Download zip to computer". Zip'i indir/ klasörüne açın, örn.:
           indir/gokhisarmain/data.yaml, indir/gokhisarmain/train/images ...
    2) python veri_hazirla.py indir/gokhisarmain indir/balloon-detect

Çıktı: veri/balon/{train,val,test}/{images,labels} ve veri/balon/data.yaml

Sınıf eşleme (sınıf adına göre, otomatik):
    "red" / "kirmizi" / "kırmızı"  -> red_balloon
    "blue" / "mavi"                -> blue_balloon
    renksiz "balloon" / "balon"    -> kutunun içindeki renge (HSV) bakılır;
                                      kırmızı ya da mavi baskın değilse atlanır
    diğer her şey (ör. "i")        -> atlanır
Bir görüntüde atlanan tek bir kutu bile varsa o görüntü hiç alınmaz.
"""

import argparse
import hashlib
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

SINIFLAR = ["red_balloon", "blue_balloon"]
RESIM_UZANTI = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ESLEME = {"train": "train", "valid": "val", "val": "val", "test": "test"}


def sinif_esle(ad):
    """Kaynak sınıf adını hedef sınıf numarasına çevirir.
    0/1: kesin sınıf, 'renk': rengine bakılacak, None: atla."""
    ad = ad.lower()
    if "red" in ad or "kirmizi" in ad or "kırmızı" in ad:
        return 0
    if "blue" in ad or "mavi" in ad:
        return 1
    if "balloon" in ad or "balon" in ad:
        return "renk"
    return None


def renk_bul(resim, cx, cy, w, h, esik=0.25):
    """Kutunun ortasındaki bölgede kırmızı/mavi piksel oranına bakar.
    Kırmızı OpenCV'de iki aralıktadır: 0-10 ve 170-180."""
    H, W = resim.shape[:2]
    # Kenarlardaki arka planı almamak için kutunun orta %60'ı
    x1 = int((cx - w * 0.3) * W); x2 = int((cx + w * 0.3) * W)
    y1 = int((cy - h * 0.3) * H); y2 = int((cy + h * 0.3) * H)
    parca = resim[max(y1, 0):max(y2, 1), max(x1, 0):max(x2, 1)]
    if parca.size == 0:
        return None
    hsv = cv2.cvtColor(parca, cv2.COLOR_BGR2HSV)
    doygun = (hsv[..., 1] > 80) & (hsv[..., 2] > 50)
    ton = hsv[..., 0]
    kirmizi = (((ton <= 10) | (ton >= 170)) & doygun).mean()
    mavi = (((ton >= 95) & (ton <= 130)) & doygun).mean()
    if kirmizi > esik and kirmizi > 2 * mavi:
        return 0
    if mavi > esik and mavi > 2 * kirmizi:
        return 1
    return None


def isim_listesi(names):
    if isinstance(names, dict):
        return [names[k] for k in sorted(names)]
    return list(names)


def kaynak_isle(kaynak, cikis, gorulen, sayac):
    data = yaml.safe_load((kaynak / "data.yaml").read_text(encoding="utf-8"))
    adlar = isim_listesi(data["names"])
    esleme = {i: sinif_esle(ad) for i, ad in enumerate(adlar)}
    # Dosya adı öneki: "-Red-Blue Balloon Detection-.v3i.yolov8" -> "redbluebal"
    onek = re.sub(r"[^a-z0-9]", "", kaynak.name.lower())[:10] or "set"
    print(f"\n[{kaynak.name}] sınıflar:")
    for i, ad in enumerate(adlar):
        hedef = esleme[i]
        hedef_ad = SINIFLAR[hedef] if isinstance(hedef, int) else ("renge göre" if hedef else "ATLANDI")
        print(f"   {i}: {ad!r:20} -> {hedef_ad}")

    for split_klasor in kaynak.iterdir():
        split = SPLIT_ESLEME.get(split_klasor.name)
        if split is None or not (split_klasor / "images").is_dir():
            continue
        for resim_yolu in (split_klasor / "images").iterdir():
            if resim_yolu.suffix.lower() not in RESIM_UZANTI:
                continue
            ham = resim_yolu.read_bytes()
            ozet = hashlib.md5(ham).hexdigest()
            if ozet in gorulen:          # aynı görüntü başka bir sette de varsa bir kez al
                sayac["tekrar"] += 1
                continue
            gorulen.add(ozet)

            etiket_yolu = split_klasor / "labels" / (resim_yolu.stem + ".txt")
            satirlar = etiket_yolu.read_text().split("\n") if etiket_yolu.exists() else []

            resim = None
            yeni = []
            atlandi = False
            for satir in satirlar:
                parca = satir.split()
                if len(parca) != 5:      # boş satır veya poligon etiket
                    continue
                sinif = esleme.get(int(parca[0]))
                cx, cy, w, h = map(float, parca[1:])
                if sinif == "renk":
                    if resim is None:
                        resim = cv2.imdecode(np.frombuffer(ham, np.uint8), cv2.IMREAD_COLOR)
                    sinif = renk_bul(resim, cx, cy, w, h)
                    sayac["renk_" + ("belirsiz" if sinif is None else SINIFLAR[sinif])] += 1
                if sinif is None:
                    atlandi = True
                    break
                yeni.append(f"{sinif} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            # Bir kutu bile atlanırsa görüntünün tamamı atlanır. Yoksa etiketsiz kalan
            # balon, modele "bu balon değil" diye öğretilmiş olur.
            if atlandi:
                sayac["atlanan_goruntu"] += 1
                continue
            for e in yeni:
                sayac[SINIFLAR[int(e[0])]] += 1

            ad = f"{onek}_{resim_yolu.stem}"
            (cikis / split / "images").mkdir(parents=True, exist_ok=True)
            (cikis / split / "labels").mkdir(parents=True, exist_ok=True)
            shutil.copyfile(resim_yolu, cikis / split / "images" / (ad + resim_yolu.suffix.lower()))
            # Etiketsiz görüntü de kopyalanır: model "burada balon yok" demeyi öğrenir
            (cikis / split / "labels" / (ad + ".txt")).write_text("\n".join(yeni))
            sayac[split] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kaynaklar", nargs="+", help="Roboflow YOLOv8 export klasörleri (data.yaml içeren)")
    ap.add_argument("--cikis", default="veri/balon")
    args = ap.parse_args()

    cikis = Path(args.cikis)
    if cikis.exists():
        shutil.rmtree(cikis)

    from collections import Counter
    sayac = Counter()
    gorulen = set()
    for k in args.kaynaklar:
        kaynak_isle(Path(k), cikis, gorulen, sayac)

    # Kaynakta val veya test yoksa train'den ayırmak yerine uyar
    for split in ("train", "val"):
        if sayac[split] == 0:
            print(f"\nUYARI: '{split}' bölümü boş. Roboflow'da indirirken train/valid/test ayrımı olan sürümü seçin.")

    data = {
        "path": str(cikis.resolve()),
        "train": "train/images",
        "val": "val/images",
        "names": dict(enumerate(SINIFLAR)),
    }
    if sayac["test"]:
        data["test"] = "test/images"
    (cikis / "data.yaml").write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")

    print("\nÖzet")
    print(f"   görüntü: train {sayac['train']}, val {sayac['val']}, test {sayac['test']} "
          f"(tekrar eden {sayac['tekrar']}, belirsiz/bilinmeyen kutulu {sayac['atlanan_goruntu']} görüntü atlandı)")
    print(f"   kutu:    red_balloon {sayac['red_balloon']}, blue_balloon {sayac['blue_balloon']}")
    if sayac["renk_red_balloon"] or sayac["renk_blue_balloon"] or sayac["renk_belirsiz"]:
        print(f"   renksiz 'balloon' kutuları: kırmızı {sayac['renk_red_balloon']}, "
              f"mavi {sayac['renk_blue_balloon']}, belirsiz (atlandı) {sayac['renk_belirsiz']}")
    print(f"\nHazır: {cikis / 'data.yaml'}")


if __name__ == "__main__":
    main()
