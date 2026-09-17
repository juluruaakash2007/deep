"""
DeepShield — Image Preprocessor
"""

import numpy as np
from PIL import Image
import io
import torch
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def bytes_to_pil(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def preprocess_image(image_bytes: bytes) -> tuple[torch.Tensor, Image.Image]:
    """
    Returns (tensor [1,3,224,224], original PIL image)
    """
    pil_img = bytes_to_pil(image_bytes)
    tensor = _transform(pil_img).unsqueeze(0)  # add batch dim
    return tensor, pil_img


def detect_face_mediapipe(pil_img: Image.Image) -> bool:
    """
    Returns True if at least one face is detected in the image.
    Falls back to True if mediapipe not available.
    """
    try:
        import mediapipe as mp
        import numpy as np
        face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.4
        )
        img_array = np.array(pil_img)
        results = face_detection.process(img_array)
        face_detection.close()
        return bool(results.detections)
    except Exception:
        return True  # Assume face present if detection fails
