// ============================================================
// Pulsarf_ciftmotor_claude.ino
// Pulsarf_ciftmotor.ino'nun düzeltilmiş hali. balon_takip_pid.py ile birlikte çalışır.
//
// Protokol (PC -> Arduino, 115200 baud):
//   "<dy,dx>"  dy, dx = motorun ŞU ANKİ konumuna göre GÖRELİ adım sayısı (önce Y, sonra X)
//   "<0,0>"    = olduğun yerde dur
//
// Orijinale göre değişenler:
//   1) <0,0> artık motoru durduruyor. Orijinalde MultiStepper::moveTo() tüm mesafeler 0
//      olunca hiçbir şey yapmıyordu; motor bir önceki hedefe gitmeye devam ediyordu
//      (balon merkezdeyken, hedef kaybolunca ve Python kapanınca durmuyordu).
//   2) Komut zaman aşımı: PC'den KOMUT_ZAMAN_ASIMI_MS boyunca komut gelmezse
//      (USB kablosu çıktı, program dondu) motorlar kendiliğinden durur.
//   3) '<' ile '>' dışındaki karakterler yok sayılır; bozuk veri gelirse tampon
//      sınırsız büyüyüp RAM'i doldurmaz.
// Pinler, hızlar ve motor sırası orijinalle aynı.
// ============================================================

#include <AccelStepper.h>
#include <MultiStepper.h>

// PIN ÇAKIŞMASINI ÖNLEMEK İÇİN PINLERİ DÜZELTTİK:
#define DIR_Y 2
#define STEP_Y 5
#define DIR_X 4
#define STEP_X 3
#define IF_TYPE 1   // AccelStepper::DRIVER (STEP/DIR sürücü)

// adım/sn. Motor yavaş tepki veriyorsa önce bunları artırın (bkz. Python'daki PID notları).
const float MAX_SPEED_X = 30;
const float MAX_SPEED_Y = 25;
// Not: MultiStepper ivme desteklemez, motorlar sabit MAX_SPEED ile döner.
// Bu iki değer şu an etkisiz; ileride MultiStepper yerine her motor için ayrı
// run() kullanılırsa işe yarar.
const float ACCEL_X = 800;
const float ACCEL_Y = 500;

// Bu kadar ms komut gelmezse dur. 0 yazarsanız zaman aşımı kapanır.
// Python takip ederken her karede komut yollar; merkezdeyken yollamaz ama
// o zaman zaten '<0,0>' ile durmuştur.
const unsigned long KOMUT_ZAMAN_ASIMI_MS = 1000;
const unsigned int MAX_TAMPON = 24;   // "<-2147483648,-2147483648>" bile sığar

AccelStepper mX(IF_TYPE, STEP_X, DIR_X);
AccelStepper mY(IF_TYPE, STEP_Y, DIR_Y);

MultiStepper steppers;

String buf = "";
bool okuyor = false;            // '<' görüldü, '>' bekleniyor
unsigned long sonKomut = 0;     // son geçerli komutun zamanı (0 = bekleyen hareket yok)

void dur() {
  // setCurrentPosition: hedefi mevcut konuma çeker ve hızı 0 yapar -> motor bu adımda durur.
  mY.setCurrentPosition(mY.currentPosition());
  mX.setCurrentPosition(mX.currentPosition());
}

void komutIsle() {
  int k = buf.indexOf(',');
  if (k <= 0) {
    return;                     // virgül yok / başta: bozuk komut
  }

  long dy = buf.substring(0, k).toInt();
  long dx = buf.substring(k + 1).toInt();
  sonKomut = millis();

  if (dy == 0 && dx == 0) {
    dur();                      // MultiStepper sıfır hareketi yok saydığı için elle durdur
    return;
  }

  // Sıra addStepper sırasıyla aynı olmalı: önce Y, sonra X
  long tgt[2] = {
    mY.currentPosition() + dy,
    mX.currentPosition() + dx
  };

  steppers.moveTo(tgt);
}

void setup() {
  Serial.begin(115200);

  mX.setMaxSpeed(MAX_SPEED_X);
  mX.setAcceleration(ACCEL_X);

  mY.setMaxSpeed(MAX_SPEED_Y);
  mY.setAcceleration(ACCEL_Y);

  steppers.addStepper(mY);      // tgt[0]
  steppers.addStepper(mX);      // tgt[1]
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();

    if (c == '<') {
      buf = "";
      okuyor = true;
    }
    else if (c == '>') {
      if (okuyor) {
        komutIsle();
      }
      buf = "";
      okuyor = false;
    }
    else if (okuyor) {
      buf += c;
      if (buf.length() > MAX_TAMPON) {   // '>' hiç gelmedi: bozuk veri, at
        buf = "";
        okuyor = false;
      }
    }
  }

  // Güvenlik: PC sustuysa dur (bir kez)
  if (KOMUT_ZAMAN_ASIMI_MS > 0 && sonKomut != 0 &&
      millis() - sonKomut > KOMUT_ZAMAN_ASIMI_MS) {
    dur();
    sonKomut = 0;
  }

  // Motorların hedefe ulaşabilmesi için her döngüde çağrılmalı
  steppers.run();
}
