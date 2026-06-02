# 🎯 YOLO26 Real-Time Person Counter

Yapay zeka tabanlı, tarayıcı üzerinden çalışan **gerçek zamanlı insan tespiti ve sayım projesi**. Bu proje, son derece popüler olan YOLOv8 (rebranded as **YOLO26**) mimarisini kullanarak canlı kamera akışlarında insanları tespit eder, onları şık sınır kutuları (bounding box) içine alır, anlık kişi sayısını ve toplam benzersiz insan sayısını hesaplar.

Streamlit ve WebRTC altyapısı sayesinde herhangi bir harici yazılıma ihtiyaç duymadan doğrudan web tarayıcınız üzerinden çalışır.

---

## ✨ Özellikler

*   🎯 **YOLO26 Nano Model Entegrasyonu:** Yüksek doğrulukta, son derece hızlı gerçek zamanlı nesne tespiti.
*   👥 **Centroid Tracking (Nesne Takip):** Kişileri benzersiz ID'ler ile takip eder. Kişiler kadrajdan anlık çıkıp girdiğinde veya hareket ettiğinde akıllı takip mekanizması sayesinde tekrar sayılması önlenir.
*   🛡️ **Çift Sayım Engelleme (IoU NMS):** Bir kişi için birden fazla kutu çizilmesini önlemek amacıyla kesişim alanı (IoU) hesaplaması ile çakışan kutuları tekilleştirir.
*   ⏱️ **Doğrulama (Confirmation) Mekanizması:** Anlık yanlış pozitifleri önlemek için bir kişinin toplam sayıma eklenebilmesi için en az 3 ardışık frame boyunca doğrulanması gerekir.
*   🎨 **Premium Glassmorphism Arayüzü:** Streamlit üzerinde özel CSS stil şablonları, şık gradyanlar, canlı durum göstergeleri (LIVE) ve özel tasarlanmış metrik kartları.
*   🌐 **WebRTC Desteği:** Kamera görüntüsünü sunucu yerine doğrudan kullanıcının tarayıcısında işleyerek sıfır gecikme ve yüksek gizlilik sunar.
*   🎛️ **Ayarlanabilir Hassasiyet:** Arayüz üzerinden canlı olarak modelin güven eşiğini (Confidence Threshold) değiştirebilme.

---

## 📁 Proje Yapısı

```bash
realtime/
│
├── app.py                  # Canlı Streamlit Web Uygulaması (Ana Çalışma Noktası)
├── person_counter.py       # Standart OpenCV ile çalışan masaüstü uygulaması
├── requirements.txt        # Gerekli kütüphaneler listesi
└── README.md               # Proje dökümantasyonu (Şu an okuduğunuz dosya)
```

---

## 🚀 Hızlı Başlangıç

### 1. Gereksinimleri Yükleme

İlk olarak terminal veya komut satırını açıp proje dizinine gidin ve gerekli kütüphaneleri yükleyin:

```bash
pip install -r requirements.txt
```

*Alternatif olarak manuel kurulum için:*
```bash
pip install ultralytics opencv-python numpy scipy streamlit streamlit-webrtc av
```

### 2. Streamlit Web Uygulamasını Başlatma

Aşağıdaki komutla modern web arayüzünü çalıştırabilirsiniz:

```bash
python -m streamlit run app.py
```

Uygulama başarıyla başladığında otomatik olarak tarayıcınızda açılacaktır. Açılmazsa şu adrese gidin:
👉 **[http://localhost:8501](http://localhost:8501)**

### 3. Alternatif OpenCV Masaüstü Uygulamasını Başlatma

Eğer web arayüzü yerine doğrudan yerel masaüstü penceresinde çalıştırmak isterseniz:

```bash
python person_counter.py
```

*   **Çıkış:** Klavyeden `q` tuşuna basın.
*   **Duraklat/Devam Et:** Klavyeden `p` tuşuna basın.

---

## ⚙️ Akıllı Takip & Sayma Mekanizması Nasıl Çalışır?

Proje, basit bir nesne tespitinin ötesinde kararlı bir sayım performansı sunmak için **3 kademeli doğrulama** kullanır:

1.  **IoU Deduplication:** YOLO modeli bazen bir insanı iki farklı boyutlu kutu ile algılayabilir. Geliştirdiğimiz NMS (Non-Maximum Suppression) benzeri IoU filtresi, birbirinin içine giren kutuları tek bir kutuya indirger.
2.  **Mesafe Tabanlı Eşleştirme (Centroid Distance):** Her frame'deki insanların ağırlık merkezleri (centroid) ile bir önceki frame'deki centroid'ler arasındaki Öklid mesafesi hesaplanır. `max_distance=120` sınırları dahilindeki en yakın noktalar aynı kişi olarak eşleştirilir.
3.  **Onay Gecikmesi (Confirmation Threshold):** Kamera açıldığında arkadan geçen veya kameranın anlık insan sandığı gölgeler/nesneler toplam sayımı bozmasın diye, bir adayın resmi olarak "Kişi" kabul edilip ID alabilmesi için arka arkaya 3 frame boyunca kamerada kalması gerekir.

---

## 🛠️ Kullanılan Teknolojiler

*   **YOLOv8 (Ultralytics):** Nesne tespiti motoru.
*   **Streamlit & streamlit-webrtc:** Web arayüzü ve tarayıcı kamerası entegrasyonu.
*   **OpenCV:** Görüntü işleme ve görselleştirme.
*   **SciPy & NumPy:** Centroid uzaklık matris hesaplamaları.
*   **PyAV:** Gerçek zamanlı video frame dönüştürme işlemleri.

---

## 🔒 Güvenlik ve Gizlilik

Streamlit WebRTC entegrasyonu sayesinde kamera verileriniz hiçbir sunucuya **gönderilmez**. Görüntü işleme tamamen kendi bilgisayarınızda (istemci tarafında) lokal olarak gerçekleştirilir. Verileriniz tamamen güvendedir.

---

## 📝 Lisans

Bu proje eğitim ve geliştirme amaçlı hazırlanmıştır. Özgürce değiştirebilir, dağıtabilir veya kendi projelerinizde kullanabilirsiniz.
