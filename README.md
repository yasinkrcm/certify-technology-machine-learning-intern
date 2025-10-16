<h2>Certify Technology Makine Öğrenimi Stajı — Projeler</h2>

<p>
  Bu depo, staj kapsamındaki görevlerin çalıştırılabilir çözümlerini ve çıktılarının üretilmesini sağlar.
  Ana çalışma dosyası <code>Compulsory_Internship_Tasks.ipynb</code> olup veriler <code>data/</code> klasöründe yer alır.
  Aşağıdaki tüm içerik HTML olarak yazılmıştır.
  <br />
  Not: README dosyaları GitHub ve çoğu platformda HTML'yi destekler.
  Markdown ile birlikte HTML kullanımı yaygındır.
  Bu dosyada yalnızca HTML kullanılmıştır.
  
</p>

<h3>Gereksinimler</h3>
<ul>
  <li>Python 3.9+ (öneri: 3.10 veya üzeri)</li>
  <li>Jupyter Notebook ya da VS Code (Jupyter eklentisiyle)</li>
  <li>Temel kütüphaneler: <code>pandas</code>, <code>numpy</code>, <code>scikit-learn</code>, <code>matplotlib</code>, <code>seaborn</code></li>
</ul>

<h3>Nasıl Çalıştırılır</h3>
<ol>
  <li><code>Compulsory_Internship_Tasks.ipynb</code> dosyasını Jupyter veya VS Code ile açın.</li>
  <li>Hücreleri sırayla çalıştırın. Gerekli veri dosyaları <code>data/</code> altında hazırdır.</li>
  <li>Eğitim/Değerlendirme metrikleri hücre çıktılarında görünür; grafikler notebook içinde görüntülenir.</li>
</ol>

<h3>İçerik</h3>

<h4>Seviye 1 — Kodlama Alıştırmaları</h4>
<ul>
  <li>Doğrusal Regresyon (California Housing): MSE ve R²; Gerçek vs Tahmin saçılım grafiği.</li>
  <li>Karar Ağacı (Iris): doğruluk, karışıklık matrisi ve ağaç görselleştirmesi.</li>
</ul>

<h4>Seviye 1 — Zorunlu: Spam E-posta Sınıflandırma</h4>
<ul>
  <li>Yaklaşım: TF‑IDF + MultinomialNB (Naive Bayes)</li>
  <li>Metrikler: doğruluk, precision, recall, F1</li>
</ul>

<h4>Seviye 2 — Sigorta Ücreti Tahmini</h4>
<ul>
  <li>Modeller: Linear Regression, RandomForest, (isteğe bağlı) XGBoost</li>
  <li>Metrikler: MSE, R²</li>
  <li>Görselleştirmeler: korelasyon ısı haritası, Gerçek vs Tahmin, özellik önemleri</li>
</ul>

<h4>Seviye 2 — Perakende Müşteri Bölümleme (K‑Means)</h4>
<ul>
  <li>Yöntem: Dirsek yöntemi ile k belirleme, ardından K‑Means; PCA ile 2B küme görselleştirme</li>
</ul>

<h4>Seviye 3 — Kredi Kartı Sahtecilik Tespiti</h4>
<ul>
  <li>Modeller: Lojistik Regresyon, RandomForest (uygun ölçekleme/dengeleme ile)</li>
  <li>Metrikler: ROC‑AUC, PR‑AUC</li>
  <li>Görselleştirmeler: ROC ve Precision‑Recall eğrileri</li>
</ul>

<h3>Notlar</h3>
<ul>
  <li>Veri setleri <code>data/</code> altında bulunduğu için çevrimdışı çalışma mümkündür.</li>
  <li>Grafik ve ara çıktılar her yeniden çalıştırmada tekrar üretilebilir.</li>
</ul>

<h3>Çıktı Görselleri (photos/)</h3>
<div style="display:flex; flex-direction:column; align-items:center; gap:24px; margin-top:12px;">
  <figure style="margin:0; text-align:center;">
    <img src="photos/iris_confusion_matrix.png" alt="Iris Karışıklık Matrisi" style="max-width:100%; width:480px; height:auto; border:1px solid #ddd; border-radius:6px;" />
    <figcaption style="margin-top:8px; color:#666;">Iris - Karışıklık Matrisi</figcaption>
  </figure>
  <figure style="margin:0; text-align:center;">
    <img src="photos/iris_tree.png" alt="Iris Karar Ağacı" style="max-width:100%; width:640px; height:auto; border:1px solid #ddd; border-radius:6px;" />
    <figcaption style="margin-top:8px; color:#666;">Iris - Karar Ağacı</figcaption>
  </figure>
  <figure style="margin:0; text-align:center;">
    <img src="photos/spam_confusion_matrix.png" alt="Spam Karışıklık Matrisi" style="max-width:100%; width:480px; height:auto; border:1px solid #ddd; border-radius:6px;" />
    <figcaption style="margin-top:8px; color:#666;">Spam - Karışıklık Matrisi</figcaption>
  </figure>
  <figure style="margin:0; text-align:center;">
    <img src="photos/titanic_confusion_matrix.png" alt="Titanic Karışıklık Matrisi" style="max-width:100%; width:480px; height:auto; border:1px solid #ddd; border-radius:6px;" />
    <figcaption style="margin-top:8px; color:#666;">Titanic - Karışıklık Matrisi</figcaption>
  </figure>
  <figure style="margin:0; text-align:center;">
    <img src="photos/titanic_feature_importances.png" alt="Titanic Özellik Önemleri" style="max-width:100%; width:560px; height:auto; border:1px solid #ddd; border-radius:6px;" />
    <figcaption style="margin-top:8px; color:#666;">Titanic - Özellik Önemleri</figcaption>
  </figure>
  <figure style="margin:0; text-align:center;">
    <img src="photos/titanic_target_distribution.png" alt="Titanic Hedef Dağılımı" style="max-width:100%; width:420px; height:auto; border:1px solid #ddd; border-radius:6px;" />
    <figcaption style="margin-top:8px; color:#666;">Titanic - Hedef Dağılımı</figcaption>
  </figure>
  <figure style="margin:0; text-align:center;">
    <img src="photos/insurance_corr_heatmap.png" alt="Sigorta Korelasyon Isı Haritası" style="max-width:100%; width:560px; height:auto; border:1px solid #ddd; border-radius:6px;" />
    <figcaption style="margin-top:8px; color:#666;">Sigorta - Korelasyon Isı Haritası</figcaption>
  </figure>
  <figure style="margin:0; text-align:center;">
    <img src="photos/insurance_actual_vs_pred_RandomForestRegressor.png" alt="Sigorta Gerçek vs Tahmin (RandomForest)" style="max-width:100%; width:560px; height:auto; border:1px solid #ddd; border-radius:6px;" />
    <figcaption style="margin-top:8px; color:#666;">Sigorta - Gerçek vs Tahmin (RandomForest)</figcaption>
  </figure>
  <figure style="margin:0; text-align:center;">
    <img src="photos/insurance_feature_importances_RandomForestRegressor.png" alt="Sigorta Özellik Önemleri (RandomForest)" style="max-width:100%; width:560px; height:auto; border:1px solid #ddd; border-radius:6px;" />
    <figcaption style="margin-top:8px; color:#666;">Sigorta - Özellik Önemleri (RandomForest)</figcaption>
  </figure>
  <figure style="margin:0; text-align:center;">
    <img src="photos/insurance_actual_vs_pred_XGBRegressor.png" alt="Sigorta Gerçek vs Tahmin (XGBRegressor)" style="max-width:100%; width:560px; height:auto; border:1px solid #ddd; border-radius:6px;" />
    <figcaption style="margin-top:8px; color:#666;">Sigorta - Gerçek vs Tahmin (XGBRegressor)</figcaption>
  </figure>
  <figure style="margin:0; text-align:center;">
    <img src="photos/insurance_feature_importances_XGBRegressor.png" alt="Sigorta Özellik Önemleri (XGBRegressor)" style="max-width:100%; width:560px; height:auto; border:1px solid #ddd; border-radius:6px;" />
    <figcaption style="margin-top:8px; color:#666;">Sigorta - Özellik Önemleri (XGBRegressor)</figcaption>
  </figure>
  <figure style="margin:0; text-align:center;">
    <img src="photos/segmentation_elbow.png" alt="Bölümleme Dirsek Yöntemi" style="max-width:100%; width:520px; height:auto; border:1px solid #ddd; border-radius:6px;" />
    <figcaption style="margin-top:8px; color:#666;">Perakende Bölümleme - Dirsek Yöntemi</figcaption>
  </figure>
  <figure style="margin:0; text-align:center;">
    <img src="photos/segmentation_pca_clusters.png" alt="Bölümleme PCA Küme Görselleştirme" style="max-width:100%; width:640px; height:auto; border:1px solid #ddd; border-radius:6px;" />
    <figcaption style="margin-top:8px; color:#666;">Perakende Bölümleme - PCA Küme Görselleştirme</figcaption>
  </figure>
  <figure style="margin:0; text-align:center;">
    <img src="photos/fraud_roc_curves.png" alt="Sahtecilik ROC Eğrileri" style="max-width:100%; width:560px; height:auto; border:1px solid #ddd; border-radius:6px;" />
    <figcaption style="margin-top:8px; color:#666;">Kredi Kartı Sahtecilik - ROC Eğrileri</figcaption>
  </figure>
  <figure style="margin:0; text-align:center;">
    <img src="photos/fraud_pr_curves.png" alt="Sahtecilik Precision-Recall Eğrileri" style="max-width:100%; width:560px; height:auto; border:1px solid #ddd; border-radius:6px;" />
    <figcaption style="margin-top:8px; color:#666;">Kredi Kartı Sahtecilik - Precision-Recall Eğrileri</figcaption>
  </figure>
</div>