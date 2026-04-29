# Bu sınıf sadece kamera verisini alacak ve main sınıfına yönlendirecektir
# Bu sınıfın kullanılmasınn temel nedeni birden çok kod parçası aynı anda kamerya erişmeye çalıştığında
# kaynak çakışmasının önüne geçilmesidir.
# Görüntü bu sınıfta alınır ve diğer sınıflara dağıtılır. Bu dağıtım sırasında görüntü pointerı
# kullanıldığı için zaman kaybı minimumdur

