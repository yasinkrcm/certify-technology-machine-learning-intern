## Certify Technology Makine Öğrenimi Stajı — Projeler

Bu depo, staj kapsamındaki görevlerin çalıştırılabilir çözümlerini ve çıktılarının üretilmesini sağlar. Ana çalışma dosyası `Compulsory_Internship_Tasks.ipynb` olup veriler `data/` klasöründe yer alır.

### Gereksinimler
- Python 3.9+ (öneri: 3.10 veya üzeri)
- Jupyter Notebook ya da VS Code (Jupyter eklentisiyle)
- Temel kütüphaneler: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`

### Nasıl Çalıştırılır
1. `Compulsory_Internship_Tasks.ipynb` dosyasını Jupyter veya VS Code ile açın.
2. Hücreleri sırayla çalıştırın. Gerekli veri dosyaları `data/` altında hazırdır.
3. Eğitim/Değerlendirme metrikleri hücre çıktılarında görünür; grafikler notebook içinde görüntülenir.

### İçerik

#### Seviye 1 — Kodlama Alıştırmaları
- Doğrusal Regresyon (California Housing): MSE ve R²; Gerçek vs Tahmin saçılım grafiği.
- Karar Ağacı (Iris): doğruluk, karışıklık matrisi ve ağaç görselleştirmesi.

#### Seviye 1 — Zorunlu: Spam E-posta Sınıflandırma
- Yaklaşım: TF‑IDF + MultinomialNB (Naive Bayes)
- Metrikler: doğruluk, precision, recall, F1

#### Seviye 2 — Sigorta Ücreti Tahmini
- Modeller: Linear Regression, RandomForest, (isteğe bağlı) XGBoost
- Metrikler: MSE, R²
- Görselleştirmeler: korelasyon ısı haritası, Gerçek vs Tahmin, özellik önemleri

#### Seviye 2 — Perakende Müşteri Bölümleme (K‑Means)
- Yöntem: Dirsek yöntemi ile k belirleme, ardından K‑Means; PCA ile 2B küme görselleştirme

#### Seviye 3 — Kredi Kartı Sahtecilik Tespiti
- Modeller: Lojistik Regresyon, RandomForest (uygun ölçekleme/dengeleme ile)
- Metrikler: ROC‑AUC, PR‑AUC
- Görselleştirmeler: ROC ve Precision‑Recall eğrileri

### Notlar
- Veri setleri `data/` altında bulunduğu için çevrimdışı çalışma mümkündür.
- Grafik ve ara çıktılar her yeniden çalıştırmada tekrar üretilebilir.
