# Digital Music Platforms - Churn Prediction

Bu proje, dijital müzik platformlarında kullanıcıların abonelik iptali davranışlarını tahmin etmek amacıyla hazırlanmıştır. Çalışmada KKBOX veri seti kullanılarak kullanıcı bilgileri, işlem kayıtları ve dinleme davranışları birleştirilmiş; ardından farklı makine öğrenmesi modelleri ile abonelik iptali tahmini gerçekleştirilmiştir.

## Projenin Amacı

Dijital müzik platformları abonelik temelli gelir modeliyle çalıştığı için kullanıcıların aboneliklerini sürdürmesi platformların sürdürülebilirliği açısından kritik öneme sahiptir. Kullanıcıların aboneliklerini iptal etmesi yalnızca doğrudan gelir kaybına değil, aynı zamanda müşteri bağlılığının azalmasına ve yeni kullanıcı kazanım maliyetlerinin artmasına da neden olabilir.

Bu projenin temel amacı, kullanıcıların geçmiş davranışlarından yararlanarak abonelik iptali riski taşıyan kullanıcıları önceden tahmin edebilen bir makine öğrenmesi süreci geliştirmektir.

## Kullanılan Veri Seti

Projede KKBOX Churn Prediction veri seti kullanılmıştır. Veri setinde kullanıcı kimlikleri, abonelik durumu, üyelik bilgileri, ödeme/işlem kayıtları ve dinleme davranışlarına ilişkin bilgiler bulunmaktadır.

Kullanılan temel dosyalar:

- `train.csv`: Kullanıcı kimliği ve abonelik iptali bilgisini içerir.
- `members_v3.csv`: Kullanıcıların üyelik bilgilerini içerir.
- `transactions.csv`: Kullanıcıların ödeme ve abonelik işlem kayıtlarını içerir.
- `user_logs.csv`: Kullanıcıların dinleme davranışlarına ilişkin kayıtları içerir.

Hedef değişken:

- `is_churn = 1`: Kullanıcı aboneliğini iptal etmiştir.
- `is_churn = 0`: Kullanıcı aboneliğini devam ettirmiştir.

## Proje Akışı

Proje üç temel Python dosyasından oluşmaktadır:

### 1. Veri Ön İşleme ve Birleştirme

`pipeline.py` dosyası ile veri setleri okunur, büyük dosyalar parça parça işlenir, kullanıcı bazlı özet değişkenler oluşturulur ve modellemeye hazır veri setleri üretilir.

Bu aşamada yapılan işlemler:

- `train.csv`, `members_v3.csv`, `transactions.csv` ve `user_logs.csv` dosyalarının okunması
- Büyük veri dosyalarının chunk yöntemiyle işlenmesi
- Kullanıcı bazlı işlem özetlerinin çıkarılması
- Kullanıcı bazlı dinleme davranışı özetlerinin çıkarılması
- Eksik verilerin düzenlenmesi
- Tarih değişkenlerinden yıl/ay gibi yeni özelliklerin çıkarılması
- Ödeme farkı, dinleme yoğunluğu ve tam dinleme oranı gibi yeni özelliklerin oluşturulması
- Eğitim ve test veri setlerinin oluşturulması

Üretilen temel dosyalar:

- `transactions_reduced.csv`
- `user_logs_reduced.csv`
- `merged_preprocessed_full.csv`
- `X_train.csv`
- `X_test.csv`
- `y_train.csv`
- `y_test.csv`
- `preview_1000_rows.csv`

### 2. Model Eğitimi ve Değerlendirme

`Create-models.py` dosyası ile farklı makine öğrenmesi modelleri eğitilir ve test verisi üzerinde karşılaştırılır.

Kullanılan modeller:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- LightGBM

Model performansları aşağıdaki metriklerle değerlendirilmiştir:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Model sonuçları `model_results.csv` dosyasına kaydedilir.

### 3. Sonuçların Görselleştirilmesi

`plot.py` dosyası ile modellerin F1 skorları karşılaştırmalı olarak grafikleştirilir.

Üretilen çıktı:

- `f1_comparison.png`

## Kurulum

Projeyi çalıştırmak için öncelikle gerekli Python kütüphanelerinin kurulması gerekir.
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib


Kullanım

Öncelikle veri seti dosyalarının proje içinde uygun klasöre yerleştirilmesi gerekir. Daha sonra sırasıyla aşağıdaki dosyalar çalıştırılır.

1. Veri hazırlama
python pipeline.py

Bu işlem sonucunda modellemeye hazır eğitim ve test dosyaları oluşturulur.

2. Model eğitimi
python Create-models.py

Bu işlem sonucunda modeller eğitilir, performans metrikleri hesaplanır ve model_results.csv dosyası oluşturulur.

3. Grafik oluşturma
python plot.py

Bu işlem sonucunda modellerin F1 skorlarını karşılaştıran grafik oluşturulur.

Proje Yapısı
DigitalMusicPlatforms---Churn/
│
├── pipeline.py
├── Create-models.py
├── plot.py
│
├── X_train.csv
├── X_test.csv
├── y_train.csv
├── y_test.csv
│
├── model_results.csv
└── f1_comparison.png

Not: Büyük veri dosyaları GitHub’a yüklenmemiş olabilir. Bu durumda KKBOX veri seti ayrıca indirilerek ilgili klasöre yerleştirilmelidir.

Özellik Mühendisliği

Projede yalnızca ham veri kullanılmamış, kullanıcı davranışlarını daha anlamlı temsil edebilmek için yeni değişkenler oluşturulmuştur. Bunlardan bazıları şunlardır:

Kullanıcının toplam işlem sayısı
Farklı ödeme yöntemi sayısı
Ortalama ödeme planı süresi
Ortalama ödenen tutar
Otomatik yenileme oranı
İptal işlem oranı
Son işlem tarihi ve üyelik bitiş tarihi arasındaki gün farkı
Toplam dinleme süresi
Eşsiz şarkı sayısı
Tam dinleme oranı
Şarkı başına ortalama dinleme süresi
İndirim miktarı

Bu değişkenler, kullanıcının platformla kurduğu ekonomik ve davranışsal ilişkiyi daha güçlü biçimde temsil etmek için kullanılmıştır.

Değerlendirme

Bu projede abonelik iptali tahmini bir ikili sınıflandırma problemi olarak ele alınmıştır. Farklı makine öğrenmesi algoritmaları aynı eğitim ve test verileri üzerinde değerlendirilmiş, böylece modellerin performansları karşılaştırmalı olarak incelenmiştir.

Özellikle F1 Score metriği, abonelik iptali tahmini gibi dengesiz sınıf dağılımı görülebilecek problemlerde önemli bir ölçüt olarak değerlendirilmiştir. ROC-AUC metriği ise modelin sınıfları ayırt etme başarısını ölçmek için kullanılmıştır.

Akademik Bağlam

Bu proje, dijital müzik platformlarında kullanıcı kaybı problemini makine öğrenmesi yöntemleriyle incelemektedir. Çalışma, kullanıcı davranışlarının yalnızca geçmiş abonelik durumuyla değil; ödeme alışkanlıkları, işlem geçmişi ve platform kullanım yoğunluğu gibi çok boyutlu değişkenlerle birlikte ele alınması gerektiğini göstermeyi amaçlamaktadır.
