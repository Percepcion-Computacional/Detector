import cv2
import asyncio
import numpy as np
from ultralytics import YOLO
from utils.image_processing import decode_base64_image, encode_image_base64
from models.schemas import Detection, FrameResponse
from typing import List, Dict
from collections import defaultdict, deque
from datetime import datetime
import os
import base64

# ---------------------------------------------------------------------------
# Directorio donde se guardan los snapshots de detecciones de armas
# ---------------------------------------------------------------------------
SNAPSHOTS_DIR = "snapshots"
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Umbrales de confianza POR CLASE
# ---------------------------------------------------------------------------
DEFAULT_CONFIDENCE_THRESHOLD = 0.45

WEAPON_KEYWORDS = {"gun", "pistol", "rifle", "knife", "weapon", "arma", "pistola", "cuchillo"}
PERSON_KEYWORDS  = {"person", "persona", "human"}

WEAPON_CONFIDENCE_THRESHOLD = 0.40
PERSON_CONFIDENCE_THRESHOLD = 0.40

# Si un arma se detecta sobre una persona, el recuadro real del arma es diminuto
# en comparación al cuerpo (IoU < 0.05). Si el IoU es mayor a 0.30, significa
# que el modelo dibujó el recuadro del arma del tamaño de casi toda la persona.
OVERLAP_IOU_THRESHOLD    = 0.30

# Temporal smoothing (1 significa desactivado, muestra todo de inmediato)
SMOOTHING_WINDOW   = 5
SMOOTHING_MIN_HITS = 1

# Cooldown mínimo entre snapshots (segundos) para no guardar uno por frame
SNAPSHOT_COOLDOWN_SECONDS = 3.0

# Colores (BGR)
WEAPON_COLOR  = (0,   0, 255)
PERSON_COLOR  = (255, 200, 0)
DEFAULT_COLOR = (0, 165, 255)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _iou(boxA: List[int], boxB: List[int]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter)


def _classify_name(name: str) -> str:
    name_lower = name.lower()
    if any(k in name_lower for k in WEAPON_KEYWORDS):
        return "weapon"
    if any(k in name_lower for k in PERSON_KEYWORDS):
        return "person"
    return "other"


def _get_color(class_type: str) -> tuple:
    if class_type == "weapon":
        return WEAPON_COLOR
    if class_type == "person":
        return PERSON_COLOR
    return DEFAULT_COLOR


def _get_threshold(class_type: str) -> float:
    if class_type == "weapon":
        return WEAPON_CONFIDENCE_THRESHOLD
    if class_type == "person":
        return PERSON_CONFIDENCE_THRESHOLD
    return DEFAULT_CONFIDENCE_THRESHOLD


# ---------------------------------------------------------------------------
# Guardar snapshot en disco
# ---------------------------------------------------------------------------
def _save_snapshot(img, weapon_detections: List[dict]) -> dict | None:
    """
    Guarda el frame anotado como JPEG en SNAPSHOTS_DIR.
    Retorna metadata del snapshot o None si falla.
    """
    try:
        ts = datetime.now()
        filename = ts.strftime("snapshot_%Y%m%d_%H%M%S_%f") + ".jpg"
        filepath = os.path.join(SNAPSHOTS_DIR, filename)

        cv2.imwrite(filepath, img, [cv2.IMWRITE_JPEG_QUALITY, 90])

        # Thumbnail en base64 para enviar al frontend de vuelta
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        thumbnail_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")

        weapons_info = [
            {"name": d["name"], "confidence": round(d["conf"], 4)}
            for d in weapon_detections
        ]

        print(f"📸 Snapshot guardado: {filepath} | Armas: {weapons_info}")

        return {
            "filename":  filename,
            "timestamp": ts.isoformat(),
            "weapons":   weapons_info,
            "thumbnail": thumbnail_b64,
        }
    except Exception as e:
        print(f"❌ Error guardando snapshot: {e}")
        return None


class WeaponDetector:
    def __init__(self, model_path: str = "models/best.pt"):
        try:
            self.model = YOLO(model_path)
            print(f"✅ Model loaded from: {model_path}")
            print(f"   Classes: {self.model.names}")
        except Exception as e:
            print(f"❌ Error loading model from {model_path}: {e}")
            self.model = None

        self._history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=SMOOTHING_WINDOW)
        )
        self._last_snapshot_time: float = 0.0

    # ------------------------------------------------------------------
    # Pipeline de filtrado
    # ------------------------------------------------------------------
    def _collect_candidates(self, boxes) -> List[dict]:
        candidates = []
        for box in boxes:
            conf = float(box.conf[0])
            cls  = int(box.cls[0])
            name = self.model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_type = _classify_name(name)
            candidates.append({
                "conf": conf, "cls": cls, "name": name,
                "class_type": class_type, "box": [x1, y1, x2, y2],
            })
        return candidates

    def _apply_confidence_filter(self, candidates: List[dict]) -> List[dict]:
        return [c for c in candidates if c["conf"] >= _get_threshold(c["class_type"])]

    def _apply_overlap_filter(self, candidates: List[dict]) -> List[dict]:
        persons = [c for c in candidates if c["class_type"] == "person"]
        result  = []
        for c in candidates:
            if c["class_type"] != "weapon":
                result.append(c)
                continue
            
            # IoU normal
            max_iou = max((_iou(c["box"], p["box"]) for p in persons), default=0.0)
            
            # Un arma real dentro del recuadro de una persona tendrá un IoU minúsculo (< 0.05)
            # porque su área es mucho menor. Si el IoU es > 0.3, significa que el recuadro
            # del "arma" es del tamaño de casi toda la persona (falso positivo del modelo).
            if max_iou > OVERLAP_IOU_THRESHOLD:
                print(
                    f"⚠️  Descartada '{c['name']}' conf={c['conf']:.2f} "
                    f"porque el recuadro es del tamaño de una persona (IoU={max_iou:.2f})"
                )
            else:
                result.append(c)
        return result

    def _apply_temporal_smoothing(self, candidates: List[dict]) -> List[dict]:
        detected_now = {c["name"] for c in candidates}
        for name in set(self._history.keys()) | detected_now:
            self._history[name].append(name in detected_now)
        stable = {n for n, h in self._history.items() if sum(h) >= SMOOTHING_MIN_HITS}
        return [c for c in candidates if c["name"] in stable]

    # ------------------------------------------------------------------
    # Dibujar y construir respuesta
    # ------------------------------------------------------------------
    def _draw_detections(self, img, candidates: List[dict]) -> List[Detection]:
        detections: List[Detection] = []
        for c in candidates:
            x1, y1, x2, y2 = c["box"]
            conf  = c["conf"]
            name  = c["name"]
            color = _get_color(c["class_type"])
            thickness = 2 if conf < 0.70 else 3

            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

            label = f"{name}: {conf * 100:.1f}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            detections.append(Detection(
                class_name=name,
                confidence=round(conf, 4),
                box=[x1, y1, x2, y2],
            ))
        return detections

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------
    async def process_frame(self, base64_image: str) -> FrameResponse:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._process_frame_sync, base64_image)

    def _process_frame_sync(self, base64_image: str) -> FrameResponse:
        import time
        try:
            img = decode_base64_image(base64_image)
            if img is None:
                return FrameResponse(error="Invalid image format")

            detections: List[Detection] = []
            snapshot = None

            if self.model:
                results = self.model(img, stream=False, verbose=False)
                for r in results:
                    if r.boxes is None:
                        continue

                    candidates = self._collect_candidates(r.boxes)
                    candidates = self._apply_confidence_filter(candidates)
                    candidates = self._apply_overlap_filter(candidates)
                    candidates = self._apply_temporal_smoothing(candidates)

                    detections = self._draw_detections(img, candidates)

                    # ── Guardar snapshot si hay armas y pasó el cooldown ──
                    weapon_candidates = [c for c in candidates if c["class_type"] == "weapon"]
                    now = time.monotonic()
                    if weapon_candidates and (now - self._last_snapshot_time) >= SNAPSHOT_COOLDOWN_SECONDS:
                        self._last_snapshot_time = now
                        snapshot = _save_snapshot(img, weapon_candidates)

            out_base64 = encode_image_base64(img)
            return FrameResponse(
                image=out_base64,
                detections=detections,
                snapshot=snapshot,
            )

        except Exception as e:
            print(f"❌ Inference error: {e}")
            return FrameResponse(error=str(e))


# Singleton compartido por toda la aplicación
detector_service = WeaponDetector()
