from pydantic import BaseModel
from typing import List, Optional, Any


class Detection(BaseModel):
    class_name: str
    confidence: float
    box: List[int]  # [x1, y1, x2, y2]


class SnapshotInfo(BaseModel):
    filename:  str
    timestamp: str
    weapons:   List[dict]   # [{"name": str, "confidence": float}]
    thumbnail: str          # base64 data URI del frame anotado


class FrameResponse(BaseModel):
    image:      Optional[str]          = None   # base64 annotated image
    detections: List[Detection]        = []     # objetos detectados
    error:      Optional[str]          = None   # error si algo falla
    snapshot:   Optional[SnapshotInfo] = None   # snapshot guardado (solo cuando hay arma)
