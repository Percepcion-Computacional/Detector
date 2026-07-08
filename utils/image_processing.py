import cv2
import numpy as np
import base64
from typing import Optional, Tuple

def decode_base64_image(base64_str: str) -> Optional[np.ndarray]:
    """Decodes a base64 string to a cv2 image."""
    try:
        header, encoded = base64_str.split(",", 1) if "," in base64_str else ("", base64_str)
        image_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

def encode_image_base64(img: np.ndarray) -> str:
    """Encodes a cv2 image to a base64 string."""
    _, buffer = cv2.imencode('.jpg', img)
    out_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{out_base64}"
