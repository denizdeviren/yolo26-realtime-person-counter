"""
YOLO26 Real-Time Person Counter — Streamlit Web App (ONNX Runtime)
===================================================================
Tarayıcı üzerinden gerçek zamanlı insan tespiti ve sayımı.
PyTorch OLMADAN — ONNX Runtime ile çalışır (Streamlit Cloud uyumlu).

Çalıştırmak için:
    streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import av
import threading
import time
import os
import logging
from collections import OrderedDict
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import onnxruntime as ort
from scipy.spatial import distance as dist

logger = logging.getLogger(__name__)

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
# ICE Server Yapılandırması (Streamlit Cloud için TURN desteği)
# ==============================================================================
def get_ice_servers():
    """Streamlit Cloud'da WebRTC bağlantısı için ICE sunucuları."""
    return [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun3.l.google.com:19302"]},
        {"urls": ["stun:stun4.l.google.com:19302"]},
        {
            "urls": "turn:openrelay.metered.ca:80",
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
        {
            "urls": "turn:openrelay.metered.ca:443",
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
        {
            "urls": "turn:openrelay.metered.ca:443?transport=tcp",
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
    ]

# ==============================================================================
# CSS Stili
# ==============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    .stApp { font-family: 'Inter', sans-serif; }

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
        color: #e2e8f0; font-size: 2rem; font-weight: 800;
        margin: 0 0 4px 0; letter-spacing: -0.5px;
    }
    .main-header p { color: #94a3b8; font-size: 0.95rem; margin: 0; }

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
    .metric-card .icon { font-size: 1.8rem; margin-bottom: 8px; }
    .metric-card .label {
        color: #94a3b8; font-size: 0.8rem; font-weight: 500;
        text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 6px;
    }
    .metric-card .value {
        font-size: 2.4rem; font-weight: 800;
        background: linear-gradient(135deg, #6366f1, #a78bfa);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        line-height: 1;
    }
    .metric-card.current .value {
        background: linear-gradient(135deg, #10b981, #34d399);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .metric-card.total .value {
        background: linear-gradient(135deg, #f59e0b, #fbbf24);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .metric-card.fps .value {
        background: linear-gradient(135deg, #ef4444, #f87171);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }

    .video-container {
        background: linear-gradient(145deg, #0f0f23, #13132d);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 16px; padding: 16px; margin-bottom: 24px;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f23, #1a1a3e);
    }
    section[data-testid="stSidebar"] .stMarkdown h2 { color: #e2e8f0; }

    .status-badge {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 6px 14px; border-radius: 20px;
        font-size: 0.8rem; font-weight: 600;
    }
    .status-live {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .status-dot {
        width: 8px; height: 8px; border-radius: 50%; background: #10b981;
        animation: pulse-dot 2s ease-in-out infinite;
    }
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4); }
        50% { opacity: 0.7; box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
    }

    .stSlider > div > div > div { color: #6366f1; }
    div[data-testid="stMetric"] { background: transparent; }
    .stAlert { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# ONNX Runtime ile YOLOv8 Inference (PyTorch gerektirmez!)
# ==============================================================================
class YOLOv8ONNX:
    """YOLOv8 ONNX modelini çalıştırır — torch/ultralytics gerektirmez."""

    # COCO sınıf isimleri — sadece 'person' (index 0) kullanılacak
    PERSON_CLASS_ID = 0

    def __init__(self, model_path, conf_threshold=0.5, iou_threshold=0.45, input_size=416):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size

        # ONNX Runtime session
        providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, image):
        """Görüntüyü ONNX model girdisine dönüştür."""
        h, w = image.shape[:2]
        self.orig_h, self.orig_w = h, w

        # Resize (letterbox)
        scale = min(self.input_size / h, self.input_size / w)
        new_w, new_h = int(w * scale), int(h * scale)
        self.scale = scale

        resized = cv2.resize(image, (new_w, new_h))

        # Padding (letterbox)
        pad_w = (self.input_size - new_w) // 2
        pad_h = (self.input_size - new_h) // 2
        self.pad_w, self.pad_h = pad_w, pad_h

        padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        # Normalize & transpose: HWC -> CHW, BGR -> RGB, 0-255 -> 0-1
        blob = padded[:, :, ::-1].astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)
        blob = np.expand_dims(blob, axis=0)

        return blob

    def postprocess(self, output):
        """ONNX çıktısını bounding box'lara dönüştür."""
        # YOLOv8 çıktısı: (1, 84, N) — 84 = 4 (box) + 80 (classes)
        predictions = output[0]  # (1, 84, N)
        predictions = predictions[0]  # (84, N)
        predictions = predictions.T  # (N, 84)

        # Güven skoru filtresi
        class_scores = predictions[:, 4:]  # (N, 80)
        max_scores = class_scores.max(axis=1)  # (N,)
        class_ids = class_scores.argmax(axis=1)  # (N,)

        # Sadece person sınıfı ve güven eşiği
        mask = (max_scores >= self.conf_threshold) & (class_ids == self.PERSON_CLASS_ID)
        filtered = predictions[mask]
        scores = max_scores[mask]

        if len(filtered) == 0:
            return []

        # Box koordinatları (cx, cy, w, h) -> (x1, y1, x2, y2)
        boxes = filtered[:, :4]
        cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        # Letterbox padding ve scale'i geri al
        x1 = (x1 - self.pad_w) / self.scale
        y1 = (y1 - self.pad_h) / self.scale
        x2 = (x2 - self.pad_w) / self.scale
        y2 = (y2 - self.pad_h) / self.scale

        # Sınırlara kırp
        x1 = np.clip(x1, 0, self.orig_w)
        y1 = np.clip(y1, 0, self.orig_h)
        x2 = np.clip(x2, 0, self.orig_w)
        y2 = np.clip(y2, 0, self.orig_h)

        # NMS
        boxes_for_nms = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
        indices = self._nms(boxes_for_nms, scores, self.iou_threshold)

        result = []
        for i in indices:
            result.append((int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])))

        return result

    def _nms(self, boxes, scores, iou_threshold):
        """Non-Maximum Suppression."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            union = areas[i] + areas[order[1:]] - inter
            iou = inter / (union + 1e-6)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

    def detect(self, image):
        """Tam tespit pipeline'ı: preprocess -> inference -> postprocess."""
        blob = self.preprocess(image)
        outputs = self.session.run(None, {self.input_name: blob})
        return self.postprocess(outputs)


# ==============================================================================
# IoU Hesaplama (Çift Sayımı Önleme)
# ==============================================================================
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


# ==============================================================================
# Centroid Tracker (Onay Tabanlı Kayıt)
# ==============================================================================
class CentroidTracker:
    def __init__(self, max_disappeared=60, max_distance=250, confirm_frames=4):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.object_boxes = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

        self.confirm_frames = confirm_frames
        self.candidates = OrderedDict()
        self.candidate_boxes = OrderedDict()
        self.candidate_seen = OrderedDict()
        self.next_candidate_id = 0
        self.total_confirmed = 0

    def _register_candidate(self, centroid, box):
        self.candidates[self.next_candidate_id] = centroid
        self.candidate_boxes[self.next_candidate_id] = box
        self.candidate_seen[self.next_candidate_id] = 1
        self.next_candidate_id += 1

    def _promote_candidate(self, cand_id):
        if cand_id in self.candidates:
            centroid = self.candidates[cand_id]
            box = self.candidate_boxes[cand_id]
            self.objects[self.next_object_id] = centroid
            self.object_boxes[self.next_object_id] = box
            self.disappeared[self.next_object_id] = 0
            self.next_object_id += 1
            self.total_confirmed += 1
            self._deregister_candidate(cand_id)

    def _deregister_candidate(self, cand_id):
        for d in (self.candidates, self.candidate_boxes, self.candidate_seen):
            d.pop(cand_id, None)

    def _deregister(self, object_id):
        for d in (self.objects, self.object_boxes, self.disappeared):
            d.pop(object_id, None)

    def update(self, rects):
        if len(rects) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self._deregister(oid)
            for cid in list(self.candidate_seen.keys()):
                self.candidate_seen[cid] -= 1
                if self.candidate_seen[cid] <= 0:
                    self._deregister_candidate(cid)
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        input_boxes = []
        for i, (x1, y1, x2, y2) in enumerate(rects):
            input_centroids[i] = (int((x1 + x2) / 2.0), int((y1 + y2) / 2.0))
            input_boxes.append((x1, y1, x2, y2))

        matched_input = set()

        # Onaylı nesnelerle eşleştir
        if len(self.objects) > 0:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
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
                overlap = compute_iou(prev_box, current_box) > 0.05 if prev_box else False
                if D[row, col] > self.max_distance and not overlap:
                    continue
                self.objects[obj_id] = input_centroids[col]
                self.object_boxes[obj_id] = input_boxes[col]
                self.disappeared[obj_id] = 0
                used_rows.add(row)
                matched_input.add(col)

            for row in set(range(D.shape[0])).difference(used_rows):
                obj_id = object_ids[row]
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self._deregister(obj_id)

        # Adaylarla eşleştir
        unmatched = [i for i in range(len(input_centroids)) if i not in matched_input]
        if len(self.candidates) > 0 and len(unmatched) > 0:
            cand_ids = list(self.candidates.keys())
            cand_centroids = np.array(list(self.candidates.values()))
            um_centroids = input_centroids[unmatched]
            D_c = dist.cdist(cand_centroids, um_centroids)
            c_rows = D_c.min(axis=1).argsort()
            c_cols = D_c.argmin(axis=1)[c_rows]
            used_cr, used_cc = set(), set()

            for (cr, cc) in zip(c_rows, c_cols):
                if cr in used_cr or cc in used_cc:
                    continue
                cid = cand_ids[cr]
                cbox = input_boxes[unmatched[cc]]
                pbox = self.candidate_boxes.get(cid)
                overlap = compute_iou(pbox, cbox) > 0.05 if pbox else False
                if D_c[cr, cc] > self.max_distance and not overlap:
                    continue
                self.candidates[cid] = um_centroids[cc]
                self.candidate_boxes[cid] = cbox
                self.candidate_seen[cid] += 1
                used_cr.add(cr)
                used_cc.add(cc)
                matched_input.add(unmatched[cc])
                if self.candidate_seen[cid] >= self.confirm_frames:
                    self._promote_candidate(cid)

            for cr in set(range(len(cand_ids))).difference(used_cr):
                cid = cand_ids[cr]
                if cid in self.candidate_seen:
                    self.candidate_seen[cid] -= 1
                    if self.candidate_seen[cid] <= 0:
                        self._deregister_candidate(cid)

        # Yeni aday kaydı
        for idx in range(len(input_centroids)):
            if idx not in matched_input:
                nc = input_centroids[idx]
                nb = input_boxes[idx]
                dup = False
                for oid, oc in self.objects.items():
                    d = np.linalg.norm(nc - oc)
                    ob = self.object_boxes.get(oid)
                    iou = compute_iou(ob, nb) if ob else 0
                    if d < 100 or iou > 0.1:
                        dup = True
                        break
                if not dup:
                    self._register_candidate(nc, nb)

        return self.objects


# ==============================================================================
# ONNX Model Yükleme (cache ile)
# ==============================================================================
@st.cache_resource
def load_model():
    """ONNX modelini yükle."""
    model_path = os.path.join(os.path.dirname(__file__), "yolov8n.onnx")
    return YOLOv8ONNX(model_path, conf_threshold=0.5, iou_threshold=0.45, input_size=416)


# ==============================================================================
# Video İşlemci
# ==============================================================================
class PersonCounterProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = load_model()
        self.tracker = CentroidTracker(
            max_disappeared=60, max_distance=250, confirm_frames=4
        )
        self.confidence = 0.5
        self.current_count = 0
        self.total_count = 0
        self._lock = threading.Lock()
        self._frame_count = 0
        self._last_rects = []

    def draw_fancy_box(self, frame, x1, y1, x2, y2, obj_id):
        colors = [
            (99, 102, 241), (16, 185, 129), (245, 158, 11), (239, 68, 68),
            (139, 92, 246), (6, 182, 212), (236, 72, 153), (34, 197, 94),
        ]
        color = colors[obj_id % len(colors)]
        color_bgr = (color[2], color[1], color[0])
        thickness = 2
        corner_len = min(25, min(x2 - x1, y2 - y1) // 3)

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bgr, -1)
        cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 1)

        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color_bgr, thickness + 1)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color_bgr, thickness + 1)
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color_bgr, thickness + 1)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color_bgr, thickness + 1)
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color_bgr, thickness + 1)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color_bgr, thickness + 1)
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color_bgr, thickness + 1)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color_bgr, thickness + 1)

        label = f"Kisi #{obj_id}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        label_y = max(y1 - 8, th + 8)
        cv2.rectangle(frame, (x1, label_y - th - 8), (x1 + tw + 12, label_y + 4), color_bgr, -1)
        cv2.putText(frame, label, (x1 + 6, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    def draw_overlay(self, frame, current, total):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 70), (15, 15, 35), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        for i in range(w):
            ratio = i / w
            r = int(99 + (139 - 99) * ratio)
            g = int(102 + (92 - 102) * ratio)
            b = int(241 + (246 - 241) * ratio)
            cv2.line(frame, (i, 70), (i, 72), (b, g, r), 1)

        cv2.putText(frame, "YOLO26 INSAN SAYICI", (16, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (167, 139, 250), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Anlik: {current}", (16, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (52, 211, 153), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Toplam: {total}", (180, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (251, 191, 36), 1, cv2.LINE_AA)

        badge = "LIVE"
        (bw, _), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        bx = w - bw - 30
        cv2.circle(frame, (bx - 8, 30), 5, (0, 128, 0), -1)
        cv2.putText(frame, badge, (bx + 2, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self._frame_count += 1

        # Performans: her 2. frame'de tespit yap
        if self._frame_count % 2 == 0:
            self.model.conf_threshold = self.confidence
            rects = self.model.detect(img)
            objects = self.tracker.update(rects)

            with self._lock:
                self.current_count = len(objects)
                self.total_count = self.tracker.total_confirmed
                self._last_rects = rects
        else:
            with self._lock:
                rects = self._last_rects
            objects = self.tracker.objects

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
            cv2.circle(img, (int(centroid[0]), int(centroid[1])), 4, (241, 102, 99), -1)
            cv2.circle(img, (int(centroid[0]), int(centroid[1])), 6, (241, 102, 99), 1)

        self.draw_overlay(img, self.current_count, self.total_count)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ==============================================================================
# Streamlit Arayüzü
# ==============================================================================

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

with st.sidebar:
    st.markdown("## ⚙️ Ayarlar")
    st.markdown("---")
    confidence = st.slider(
        "🎯 Güven Eşiği", min_value=0.1, max_value=1.0, value=0.5, step=0.05,
        help="Düşük = daha fazla tespit, Yüksek = daha doğru"
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
    - **Model:** YOLOv8 Nano (ONNX)
    - **Sınıf:** İnsan (person)
    - **Tracking:** Centroid-based
    - **Onay:** 4 frame doğrulama
    """)
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:#64748b; font-size:0.75rem;'>"
        "YOLO26 Person Counter v2.0<br>Powered by ONNX Runtime & Streamlit</p>",
        unsafe_allow_html=True
    )

ice_servers = get_ice_servers()
col_video, col_stats = st.columns([3, 1])

with col_video:
    st.markdown('<div class="video-container">', unsafe_allow_html=True)
    ctx = webrtc_streamer(
        key="person-counter-onnx",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=PersonCounterProcessor,
        media_stream_constraints={
            "video": {"width": {"ideal": 640, "max": 1280}, "height": {"ideal": 480, "max": 720}},
            "audio": False,
        },
        rtc_configuration={"iceServers": ice_servers},
        video_receiver_size=1,
        async_processing=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)
    if ctx.video_processor:
        ctx.video_processor.confidence = confidence

with col_stats:
    current_ph = st.empty()
    total_ph = st.empty()
    status_ph = st.empty()

    current_ph.markdown("""
    <div class="metric-card current">
        <div class="icon">👥</div>
        <div class="label">Anlık Kişi Sayısı</div>
        <div class="value">0</div>
    </div>
    """, unsafe_allow_html=True)
    total_ph.markdown("""
    <div class="metric-card total">
        <div class="icon">📊</div>
        <div class="label">Toplam Sayım</div>
        <div class="value">0</div>
    </div>
    """, unsafe_allow_html=True)
    status_ph.markdown("""
    <div class="metric-card">
        <div class="icon">⏸️</div>
        <div class="label">Durum</div>
        <div class="value" style="font-size:1.2rem; background:linear-gradient(135deg,#64748b,#94a3b8);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">BEKLEMEDE</div>
    </div>
    """, unsafe_allow_html=True)

    while ctx.state.playing:
        if ctx.video_processor:
            with ctx.video_processor._lock:
                current = ctx.video_processor.current_count
                total = ctx.video_processor.total_count
            current_ph.markdown(f"""
            <div class="metric-card current">
                <div class="icon">👥</div>
                <div class="label">Anlık Kişi Sayısı</div>
                <div class="value">{current}</div>
            </div>
            """, unsafe_allow_html=True)
            total_ph.markdown(f"""
            <div class="metric-card total">
                <div class="icon">📊</div>
                <div class="label">Toplam Sayım</div>
                <div class="value">{total}</div>
            </div>
            """, unsafe_allow_html=True)
            status_ph.markdown("""
            <div class="metric-card fps">
                <div class="icon">🔴</div>
                <div class="label">Durum</div>
                <div class="value" style="font-size:1.2rem;">CANLI</div>
            </div>
            """, unsafe_allow_html=True)
        time.sleep(0.3)

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
