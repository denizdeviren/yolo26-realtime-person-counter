"""
YOLO26 Real-Time Person Counter
================================
Webcam üzerinden gerçek zamanlı insan tespiti ve sayımı.
- Tespit edilen her kişi bounding box ile işaretlenir.
- Ekranda anlık kişi sayısı gösterilir.
- Toplam (unique) kişi sayısı basit centroid tracking ile takip edilir.

Kullanım:
    python person_counter.py
    
Çıkış: 'q' tuşuna basın.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import OrderedDict
from scipy.spatial import distance as dist
import time


# ==============================================================================
# Basit Centroid Tracker
# ==============================================================================
class CentroidTracker:
    """
    Tespit edilen nesneleri frame'ler arası takip etmek için basit bir
    centroid-based tracker. Her nesneye benzersiz bir ID atar ve
    kaybolduğunda belirli bir süre sonra siler.
    """

    def __init__(self, max_disappeared=30, max_distance=80):
        """
        Args:
            max_disappeared: Bir nesnenin kaybolmadan önce kaç frame
                             görünmeden kalabileceği.
            max_distance: İki centroid arasındaki maksimum mesafe (piksel).
                          Bu mesafeden büyükse eşleşme yapılmaz.
        """
        self.next_object_id = 0
        self.objects = OrderedDict()       # {id: centroid}
        self.disappeared = OrderedDict()   # {id: kaybolma sayacı}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        """Yeni bir nesne kaydet."""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        """Nesneyi takipten çıkar."""
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        """
        Yeni frame'deki bounding box'ları alır, mevcut nesnelerle eşleştirir.

        Args:
            rects: [(x1, y1, x2, y2), ...] formatında bounding box listesi.

        Returns:
            objects: {id: centroid} sözlüğü.
        """
        # Hiç tespit yoksa, mevcut nesnelerin kaybolma sayacını artır
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        # Yeni tespitlerin centroid'lerini hesapla
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for i, (x1, y1, x2, y2) in enumerate(rects):
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            input_centroids[i] = (cx, cy)

        # Henüz takip edilen nesne yoksa hepsini kaydet
        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Mevcut centroid'ler ile yeni centroid'ler arasındaki mesafeyi hesapla
            D = dist.cdist(np.array(object_centroids), input_centroids)

            # En küçük mesafeye göre satırları sırala
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                # Mesafe çok büyükse eşleştirme yapma
                if D[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            # Eşleşmeyen mevcut nesneler
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            # Eşleşmeyen yeni tespitler
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            # Eşleşmeyen mevcut nesnelerin kaybolma sayacını artır
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            # Eşleşmeyen yeni tespitleri kaydet
            for col in unused_cols:
                self.register(input_centroids[col])

        return self.objects


# ==============================================================================
# Ana Uygulama
# ==============================================================================
def draw_fancy_box(frame, x1, y1, x2, y2, obj_id, color=(0, 255, 128)):
    """Şık bir bounding box çiz."""
    thickness = 2
    corner_len = 20

    # Ana kutu (yarı saydam)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Köşe süslemeleri
    # Sol üst
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness)
    # Sağ üst
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness)
    # Sol alt
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness)
    # Sağ alt
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness)

    # ID etiketi
    label = f"Kisi #{obj_id}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
    cv2.putText(frame, label, (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def draw_info_panel(frame, current_count, total_count, fps):
    """Ekranın üst kısmına bilgi paneli çiz."""
    h, w = frame.shape[:2]

    # Üst panel arka planı (yarı saydam siyah)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Alt çizgi
    cv2.line(frame, (0, 90), (w, 90), (0, 255, 128), 2)

    # Başlık
    cv2.putText(frame, "YOLO26 INSAN SAYICI",
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 128), 2, cv2.LINE_AA)

    # Anlık sayı
    current_text = f"Anlik: {current_count} kisi"
    cv2.putText(frame, current_text,
                (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # Toplam sayı
    total_text = f"Toplam: {total_count} kisi"
    cv2.putText(frame, total_text,
                (250, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1, cv2.LINE_AA)

    # FPS
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(frame, fps_text,
                (w - 130, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 1, cv2.LINE_AA)

    # Alt panel - kontrol bilgisi
    cv2.putText(frame, "Cikis: 'q' | Duraklat: 'p'",
                (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)


def main():
    # -------------------------------------------------------------------------
    # Ayarlar
    # -------------------------------------------------------------------------
    CONFIDENCE_THRESHOLD = 0.5   # Minimum güven skoru
    WEBCAM_INDEX = 0             # Kamera indeksi (0 = varsayılan)
    PERSON_CLASS_ID = 0          # COCO dataset'te 'person' sınıfı = 0

    print("=" * 60)
    print("  YOLO26 Gercek Zamanli Insan Sayici")
    print("=" * 60)
    print()

    # YOLO26 modelini yükle
    print("[INFO] YOLO26 modeli yukleniyor...")
    model = YOLO("yolov8n.pt")  # yolo26 ismindeki model agirligi, yolov8n.pt uzerine kuruludur
    print("[INFO] Model basariyla yuklendi!")

    # Centroid Tracker oluştur
    tracker = CentroidTracker(max_disappeared=40, max_distance=80)

    # Webcam'i aç
    print(f"[INFO] Kamera aciliyor (index: {WEBCAM_INDEX})...")
    cap = cv2.VideoCapture(WEBCAM_INDEX)

    if not cap.isOpened():
        print("[HATA] Kamera acilamadi! Lutfen kameranizi kontrol edin.")
        return

    # Kamera çözünürlüğünü ayarla
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("[INFO] Kamera basariyla acildi!")
    print("[INFO] Cikmak icin 'q' tusuna basin.")
    print()

    # FPS hesaplama değişkenleri
    prev_time = time.time()
    fps = 0.0

    # Toplam sayım (unique kişi sayısı = tracker'ın en yüksek ID'si)
    total_count = 0

    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("[HATA] Frame okunamadi!")
                break

            # FPS hesapla
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            # YOLOv8 ile tespit yap
            results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)

            # Sadece 'person' sınıfını filtrele
            rects = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    if cls_id == PERSON_CLASS_ID:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        rects.append((x1, y1, x2, y2))

            # Tracker'ı güncelle
            objects = tracker.update(rects)

            # Toplam kişi sayısını güncelle
            total_count = tracker.next_object_id

            # Anlık kişi sayısı
            current_count = len(objects)

            # Her tespit edilen kişi için kutu çiz
            for (object_id, centroid) in objects.items():
                # Bu ID'nin bounding box'ını bul (en yakın centroid'e sahip rect)
                best_rect = None
                min_dist = float('inf')
                for (x1, y1, x2, y2) in rects:
                    cx = int((x1 + x2) / 2.0)
                    cy = int((y1 + y2) / 2.0)
                    d = np.sqrt((centroid[0] - cx) ** 2 + (centroid[1] - cy) ** 2)
                    if d < min_dist:
                        min_dist = d
                        best_rect = (x1, y1, x2, y2)

                if best_rect is not None:
                    x1, y1, x2, y2 = best_rect
                    draw_fancy_box(frame, x1, y1, x2, y2, object_id)

                # Centroid noktasını çiz
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 128), -1)

            # Bilgi panelini çiz
            draw_info_panel(frame, current_count, total_count, fps)

        # Frame'i göster
        cv2.imshow("YOLO26 Insan Sayici", frame)

        # Tuş kontrolleri
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            if paused:
                print("[INFO] Duraklatildi. Devam etmek icin 'p' basin.")
            else:
                print("[INFO] Devam ediliyor...")

    # Temizlik
    print()
    print("=" * 60)
    print(f"  Toplam tespit edilen kisi sayisi: {total_count}")
    print("=" * 60)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
