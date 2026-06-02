"""
YOLO26 Real-Time Person Counter — Streamlit Web App
=====================================================
Tarayıcı üzerinden gerçek zamanlı insan tespiti ve sayımı.

Çalıştırmak için:
    streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import av
import threading
from collections import OrderedDict
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from ultralytics import YOLO
from scipy.spatial import distance as dist

# ==============================================================================
# Sayfa Ayarları
# ==============================================================================
st.set_page_config(
    page_title="YOLO26 İnsan Sayıcı",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# CSS Stili
# ==============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* Ana tema */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Başlık alanı */
    .main-header {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0d0d2b 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #6366f1, #8b5cf6, #a78bfa, #6366f1);
        background-size: 200% 100%;
        animation: shimmer 3s ease-in-out infinite;
    }
    @keyframes shimmer {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    .main-header h1 {
        color: #e2e8f0;
        font-size: 2rem;
        font-weight: 800;
        margin: 0 0 4px 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 0.95rem;
        margin: 0;
    }

    /* Metrik kartları */
    .metric-container {
        display: flex;
        gap: 16px;
        margin-bottom: 24px;
    }
    .metric-card {
        flex: 1;
        background: linear-gradient(145deg, #13132d, #1a1a3e);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 14px;
        padding: 20px 24px;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .metric-card:hover {
        border-color: rgba(99, 102, 241, 0.5);
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.15);
    }
    .metric-card .icon {
        font-size: 1.8rem;
        margin-bottom: 8px;
    }
    .metric-card .label {
        color: #94a3b8;
        font-size: 0.8rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 6px;
    }
    .metric-card .value {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1;
    }
    .metric-card.current .value {
        background: linear-gradient(135deg, #10b981, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card.total .value {
        background: linear-gradient(135deg, #f59e0b, #fbbf24);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card.fps .value {
        background: linear-gradient(135deg, #ef4444, #f87171);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Video alanı */
    .video-container {
        background: linear-gradient(145deg, #0f0f23, #13132d);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 16px;
        padding: 16px;
        margin-bottom: 24px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f23, #1a1a3e);
    }
    section[data-testid="stSidebar"] .stMarkdown h2 {
        color: #e2e8f0;
    }

    /* Status badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .status-live {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #10b981;
        animation: pulse-dot 2s ease-in-out infinite;
    }
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4); }
        50% { opacity: 0.7; box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
    }

    /* Streamlit varsayılan düzenlemeler */
    .stSlider > div > div > div { color: #6366f1; }
    div[data-testid="stMetric"] { background: transparent; }
    .stAlert { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# IoU Hesaplama & NMS (Çift Sayımı Önleme)
# ==============================================================================
def compute_iou(box1, box2):
    """İki bounding box arasındaki IoU değerini hesapla."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def remove_duplicate_boxes(rects, iou_threshold=0.4):
    """Çakışan bounding box'ları kaldır (aynı kişi için birden fazla kutuyu önle)."""
    if len(rects) <= 1:
        return rects

    keep = []
    used = set()

    for i in range(len(rects)):
        if i in used:
            continue
        keep.append(rects[i])
        for j in range(i + 1, len(rects)):
            if j in used:
                continue
            if compute_iou(rects[i], rects[j]) > iou_threshold:
                used.add(j)

    return keep


# ==============================================================================
# Centroid Tracker (İyileştirilmiş — Onay Tabanlı Kayıt)
# ==============================================================================
class CentroidTracker:
    def __init__(self, max_disappeared=60, max_distance=250, confirm_frames=4):
        self.next_object_id = 0
        self.objects = OrderedDict()       # {id: centroid} — onaylı nesneler
        self.object_boxes = OrderedDict()  # {id: bounding_box} — onaylı kutular
        self.disappeared = OrderedDict()   # {id: kaybolma sayacı}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

        # Onay mekanizması
        self.confirm_frames = confirm_frames
        self.candidates = OrderedDict()      # {temp_id: centroid}
        self.candidate_boxes = OrderedDict()  # {temp_id: box}
        self.candidate_seen = OrderedDict()   # {temp_id: kaç frame görüldü}
        self.next_candidate_id = 0
        self.total_confirmed = 0             # Doğrulanmış toplam kişi sayısı

    def _register_candidate(self, centroid, box):
        """Yeni tespiti aday olarak kaydet."""
        self.candidates[self.next_candidate_id] = centroid
        self.candidate_boxes[self.next_candidate_id] = box
        self.candidate_seen[self.next_candidate_id] = 1
        self.next_candidate_id += 1

    def _promote_candidate(self, cand_id):
        """Adayı onaylı nesne olarak terfi ettir."""
        if cand_id in self.candidates:
            centroid = self.candidates[cand_id]
            box = self.candidate_boxes[cand_id]
            self.objects[self.next_object_id] = centroid
            self.object_boxes[self.next_object_id] = box
            self.disappeared[self.next_object_id] = 0
            self.next_object_id += 1
            self.total_confirmed += 1
            self.deregister_candidate(cand_id)

    def register(self, centroid):
        """Eski API uyumu için."""
        pass

    def deregister_candidate(self, cand_id):
        """Adayı sistemden sil."""
        if cand_id in self.candidates:
            del self.candidates[cand_id]
        if cand_id in self.candidate_boxes:
            del self.candidate_boxes[cand_id]
        if cand_id in self.candidate_seen:
            del self.candidate_seen[cand_id]

    def deregister(self, object_id):
        """Onaylı nesneyi takipten çıkar."""
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.object_boxes:
            del self.object_boxes[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]

    def update(self, rects):
        # Hiç tespit yoksa, mevcut nesnelerin kaybolma sayacını artır
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            # Adayları da kademeli olarak sil
            for cand_id in list(self.candidate_seen.keys()):
                self.candidate_seen[cand_id] -= 1
                if self.candidate_seen[cand_id] <= 0:
                    self.deregister_candidate(cand_id)
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        input_boxes = []
        for i, (x1, y1, x2, y2) in enumerate(rects):
            input_centroids[i] = (int((x1 + x2) / 2.0), int((y1 + y2) / 2.0))
            input_boxes.append((x1, y1, x2, y2))

        # --- Onaylı nesnelerle eşleştir ---
        matched_input = set()

        if len(self.objects) > 0:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            # Uzaklık matrisi
            D = dist.cdist(np.array(object_centroids), input_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            used_rows = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in matched_input:
                    continue
                
                obj_id = object_ids[row]
                prev_box = self.object_boxes.get(obj_id)
                current_box = input_boxes[col]

                # Düşük FPS durumunda (Streamlit Cloud sunucusunda) hareketler hızlı görünebilir.
                # Eğer centroid uzaklığı max_distance'tan fazlaysa ama bounding box'lar 
                # hâlâ kısmen çakışıyorsa (IoU > 0.05), bunu aynı kişi olarak eşleştir!
                overlap = False
                if prev_box is not None:
                    overlap = compute_iou(prev_box, current_box) > 0.05

                if D[row, col] > self.max_distance and not overlap:
                    continue
                
                self.objects[obj_id] = input_centroids[col]
                self.object_boxes[obj_id] = input_boxes[col]
                self.disappeared[obj_id] = 0
                used_rows.add(row)
                matched_input.add(col)

            # Eşleşmeyen onaylı nesnelerin kaybolma sayacını artır
            for row in set(range(D.shape[0])).difference(used_rows):
                obj_id = object_ids[row]
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)

        # --- Adaylarla eşleştir ---
        unmatched_inputs = [i for i in range(len(input_centroids)) if i not in matched_input]

        if len(self.candidates) > 0 and len(unmatched_inputs) > 0:
            cand_ids = list(self.candidates.keys())
            cand_centroids = np.array(list(self.candidates.values()))
            unmatched_centroids = input_centroids[unmatched_inputs]

            D_cand = dist.cdist(cand_centroids, unmatched_centroids)
            c_rows = D_cand.min(axis=1).argsort()
            c_cols = D_cand.argmin(axis=1)[c_rows]
            used_c_rows, used_c_cols = set(), set()

            for (cr, cc) in zip(c_rows, c_cols):
                if cr in used_c_rows or cc in used_c_cols:
                    continue
                
                cand_id = cand_ids[cr]
                current_box = input_boxes[unmatched_inputs[cc]]
                prev_cand_box = self.candidate_boxes.get(cand_id)

                overlap = False
                if prev_cand_box is not None:
                    overlap = compute_iou(prev_cand_box, current_box) > 0.05

                if D_cand[cr, cc] > self.max_distance and not overlap:
                    continue

                self.candidates[cand_id] = unmatched_centroids[cc]
                self.candidate_boxes[cand_id] = current_box
                self.candidate_seen[cand_id] += 1
                used_c_rows.add(cr)
                used_c_cols.add(cc)
                matched_input.add(unmatched_inputs[cc])

                if self.candidate_seen[cand_id] >= self.confirm_frames:
                    self._promote_candidate(cand_id)

            # Eşleşmeyen adayların sayacını düşür
            for cr in set(range(len(cand_ids))).difference(used_c_rows):
                cand_id = cand_ids[cr]
                if cand_id in self.candidate_seen:
                    self.candidate_seen[cand_id] -= 1
                    if self.candidate_seen[cand_id] <= 0:
                        self.deregister_candidate(cand_id)

        # --- Yeni aday kaydı ---
        # AKILLI FİLTRE: Yeni tespit edilen bir kutu, hâlihazırda takip edilen 
        # onaylanmış bir kişiye çok yakınsa (d < 100px) veya onunla çakışıyorsa (IoU > 0.1),
        # mükerrer sayımı önlemek için yeni bir aday oluşturma!
        for col_idx in range(len(input_centroids)):
            if col_idx not in matched_input:
                new_centroid = input_centroids[col_idx]
                new_box = input_boxes[col_idx]
                
                is_duplicate = False
                for obj_id, obj_centroid in self.objects.items():
                    d = np.linalg.norm(new_centroid - obj_centroid)
                    obj_box = self.object_boxes.get(obj_id)
                    iou_val = compute_iou(obj_box, new_box) if obj_box else 0
                    if d < 100 or iou_val > 0.1:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    self._register_candidate(new_centroid, new_box)

        return self.objects


# ==============================================================================
# YOLOv8 Model Yükleme (cache ile)
# ==============================================================================
@st.cache_resource
def load_model(model_name):
    return YOLO(model_name)


# ==============================================================================
# Video İşlemci
# ==============================================================================
class PersonCounterProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = load_model("yolov8n.pt")
        self.tracker = CentroidTracker(
            max_disappeared=60,
            max_distance=250,
            confirm_frames=4
        )
        self.confidence = 0.5
        self.current_count = 0
        self.total_count = 0
        self._lock = threading.Lock()

    def draw_fancy_box(self, frame, x1, y1, x2, y2, obj_id):
        """Şık bounding box çiz."""
        # Renk paleti — her ID için farklı renk
        colors = [
            (99, 102, 241),   # indigo
            (16, 185, 129),   # emerald
            (245, 158, 11),   # amber
            (239, 68, 68),    # red
            (139, 92, 246),   # violet
            (6, 182, 212),    # cyan
            (236, 72, 153),   # pink
            (34, 197, 94),    # green
        ]
        color = colors[obj_id % len(colors)]
        # OpenCV kullanır BGR
        color_bgr = (color[2], color[1], color[0])

        thickness = 2
        corner_len = min(25, min(x2 - x1, y2 - y1) // 3)

        # Yarı saydam dolgu
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bgr, -1)
        cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)

        # İnce kenar çizgisi
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 1)

        # Köşe süslemeleri
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color_bgr, thickness + 1)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color_bgr, thickness + 1)
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color_bgr, thickness + 1)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color_bgr, thickness + 1)
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color_bgr, thickness + 1)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color_bgr, thickness + 1)
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color_bgr, thickness + 1)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color_bgr, thickness + 1)

        # ID etiketi
        label = f"Kisi #{obj_id}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        label_y = max(y1 - 8, th + 8)
        cv2.rectangle(frame, (x1, label_y - th - 8), (x1 + tw + 12, label_y + 4), color_bgr, -1)
        cv2.putText(frame, label, (x1 + 6, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    def draw_overlay(self, frame, current, total):
        """Ekranda bilgi paneli göster."""
        h, w = frame.shape[:2]

        # Üst panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 70), (15, 15, 35), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Gradient çizgi
        for i in range(w):
            ratio = i / w
            r = int(99 + (139 - 99) * ratio)
            g = int(102 + (92 - 102) * ratio)
            b = int(241 + (246 - 241) * ratio)
            cv2.line(frame, (i, 70), (i, 72), (b, g, r), 1)

        # Sol: başlık
        cv2.putText(frame, "YOLO26 INSAN SAYICI", (16, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (167, 139, 250), 2, cv2.LINE_AA)

        # Anlık sayı
        current_text = f"Anlik: {current}"
        cv2.putText(frame, current_text, (16, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (52, 211, 153), 1, cv2.LINE_AA)

        # Toplam sayı
        total_text = f"Toplam: {total}"
        cv2.putText(frame, total_text, (180, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (251, 191, 36), 1, cv2.LINE_AA)

        # Sağ üst: LIVE badge
        badge_text = "LIVE"
        (bw, bh), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        bx = w - bw - 30
        cv2.circle(frame, (bx - 8, 30), 5, (0, 128, 0), -1)
        cv2.putText(frame, badge_text, (bx + 2, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # YOLOv8 tespiti
        results = self.model(img, verbose=False, conf=self.confidence)

        # Sadece 'person' sınıfı (class 0)
        rects = []
        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    rects.append((x1, y1, x2, y2))

        # Çakışan kutuları kaldır (aynı kişi için birden fazla tespiti önle)
        rects = remove_duplicate_boxes(rects, iou_threshold=0.4)

        # Tracker güncelle
        objects = self.tracker.update(rects)

        with self._lock:
            self.current_count = len(objects)
            self.total_count = self.tracker.total_confirmed

        # Her kişi için kutu çiz
        for (object_id, centroid) in objects.items():
            best_rect = None
            min_d = float('inf')
            for (x1, y1, x2, y2) in rects:
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                d = np.sqrt((centroid[0] - cx) ** 2 + (centroid[1] - cy) ** 2)
                if d < min_d:
                    min_d = d
                    best_rect = (x1, y1, x2, y2)

            if best_rect:
                self.draw_fancy_box(img, *best_rect, object_id)

            # Centroid noktası
            cv2.circle(img, (int(centroid[0]), int(centroid[1])), 4, (241, 102, 99), -1)
            cv2.circle(img, (int(centroid[0]), int(centroid[1])), 6, (241, 102, 99), 1)

        # Bilgi paneli
        self.draw_overlay(img, self.current_count, self.total_count)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ==============================================================================
# Streamlit Arayüzü
# ==============================================================================

# Başlık
st.markdown("""
<div class="main-header">
    <h1>🎯 YOLO26 Gerçek Zamanlı İnsan Sayıcı</h1>
    <p>Yapay zeka destekli canlı kamera ile insan tespiti ve sayımı</p>
    <div style="margin-top: 12px;">
        <span class="status-badge status-live">
            <span class="status-dot"></span>
            Canlı Yayın Hazır
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar ayarları
with st.sidebar:
    st.markdown("## ⚙️ Ayarlar")
    st.markdown("---")

    confidence = st.slider(
        "🎯 Güven Eşiği",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Düşük değer = daha fazla tespit (daha fazla yanlış pozitif), "
             "Yüksek değer = daha az tespit (daha doğru)"
    )

    st.markdown("---")
    st.markdown("## 📖 Nasıl Kullanılır")
    st.markdown("""
    1. **START** butonuna basın
    2. Kamera izni verin
    3. Kamera açıldığında tespit başlar
    4. **STOP** ile durdurun
    """)

    st.markdown("---")
    st.markdown("## 🏷️ Model Bilgisi")
    st.markdown("""
    - **Model:** YOLO26 Nano
    - **Sınıf:** İnsan (person)
    - **Tracking:** Centroid-based
    """)

    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:#64748b; font-size:0.75rem;'>"
        "YOLO26 Person Counter v1.0<br>Powered by Ultralytics & Streamlit</p>",
        unsafe_allow_html=True
    )

# Ana içerik
col_video, col_stats = st.columns([3, 1])

with col_video:
    st.markdown('<div class="video-container">', unsafe_allow_html=True)

    ctx = webrtc_streamer(
        key="person-counter-v2",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=PersonCounterProcessor,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640, "max": 1280},
                "height": {"ideal": 480, "max": 720},
            },
            "audio": False,
        },
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        video_receiver_size=1,
        async_processing=True,
    )

    st.markdown('</div>', unsafe_allow_html=True)

    # Güven eşiğini güncelle
    if ctx.video_processor:
        ctx.video_processor.confidence = confidence

with col_stats:
    # Canlı metrikler
    current_placeholder = st.empty()
    total_placeholder = st.empty()
    status_placeholder = st.empty()

    # Metrik kartları (İlk durum / Beklemede)
    current_placeholder.markdown("""
    <div class="metric-card current">
        <div class="icon">👥</div>
        <div class="label">Anlık Kişi Sayısı</div>
        <div class="value">0</div>
    </div>
    """, unsafe_allow_html=True)

    total_placeholder.markdown("""
    <div class="metric-card total">
        <div class="icon">📊</div>
        <div class="label">Toplam Sayım</div>
        <div class="value">0</div>
    </div>
    """, unsafe_allow_html=True)

    status_placeholder.markdown("""
    <div class="metric-card">
        <div class="icon">⏸️</div>
        <div class="label">Durum</div>
        <div class="value" style="font-size:1.2rem; background:linear-gradient(135deg,#64748b,#94a3b8);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">BEKLEMEDE</div>
    </div>
    """, unsafe_allow_html=True)

    # Canlı veri güncelleme döngüsü
    import time
    while ctx.state.playing:
        if ctx.video_processor:
            with ctx.video_processor._lock:
                current = ctx.video_processor.current_count
                total = ctx.video_processor.total_count

            current_placeholder.markdown(f"""
            <div class="metric-card current">
                <div class="icon">👥</div>
                <div class="label">Anlık Kişi Sayısı</div>
                <div class="value">{current}</div>
            </div>
            """, unsafe_allow_html=True)

            total_placeholder.markdown(f"""
            <div class="metric-card total">
                <div class="icon">📊</div>
                <div class="label">Toplam Sayım</div>
                <div class="value">{total}</div>
            </div>
            """, unsafe_allow_html=True)

            status_placeholder.markdown("""
            <div class="metric-card fps">
                <div class="icon">🔴</div>
                <div class="label">Durum</div>
                <div class="value" style="font-size:1.2rem;">CANLI</div>
            </div>
            """, unsafe_allow_html=True)
        time.sleep(0.3)

# Alt bilgi
st.markdown("---")
st.markdown("""
<div style="text-align:center; padding: 16px 0;">
    <p style="color:#64748b; font-size:0.85rem;">
        🤖 <strong>YOLO26</strong> yapay zeka modeli ile desteklenmektedir &nbsp;|&nbsp;
        📷 Kamera verisi tarayıcınızda işlenir &nbsp;|&nbsp;
        🔒 Verileriniz güvendedir
    </p>
</div>
""", unsafe_allow_html=True)
