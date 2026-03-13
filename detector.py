import torch
import cv2
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
YOLO_DIR = BASE_DIR / "yolov7"

if str(YOLO_DIR) not in sys.path:
    sys.path.insert(0, str(YOLO_DIR))

from models.experimental import attempt_load
from utils.general import non_max_suppression


class PlantDiseaseDetector:
    def __init__(self, model_path, model_version='plantdoc_300_epochs3'):
        self.model_version = model_version
        self.base_path = Path(model_path)
        self.model_weights = self.base_path / 'modelo' / model_version / 'weights' / 'best.pt'

        if not self.model_weights.exists():
            raise FileNotFoundError(f"No se encontró el modelo en: {self.model_weights}")

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = attempt_load(str(self.model_weights), map_location=self.device)
        self.model.eval()

        self.class_names = [
            'Apple Scab Leaf', 'Apple leaf', 'Apple rust leaf',
            'Bell_pepper leaf spot', 'Bell_pepper leaf', 'Blueberry leaf',
            'Cherry leaf', 'Corn Gray leaf spot', 'Corn leaf blight',
            'Corn rust leaf', 'Peach leaf', 'Potato leaf late blight',
            'Potato leaf', 'Raspberry leaf', 'Soyabean leaf',
            'Squash Powdery mildew leaf', 'Strawberry leaf',
            'Tomato Early blight leaf', 'Tomato Septoria leaf spot',
            'Tomato leaf bacterial spot', 'Tomato leaf late blight',
            'Tomato leaf mosaic virus', 'Tomato leaf yellow virus',
            'Tomato leaf', 'Tomato mold leaf', 'grape leaf black rot'
        ]

    def detect(self, image_path, conf_threshold=0.15):
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"No se pudo leer la imagen en {image_path}")

            original_h, original_w = img.shape[:2]

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (640, 640))

            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            img_tensor = img_tensor.to(self.device)

            with torch.no_grad():
                pred = self.model(img_tensor)[0]
                pred = non_max_suppression(pred, conf_threshold, 0.45)

            detections = []

            if pred and pred[0] is not None and len(pred[0]) > 0:
                scale_x = original_w / 640.0
                scale_y = original_h / 640.0

                for det in pred[0]:
                    *xyxy, conf, cls = det

                    x1 = max(0, int(float(xyxy[0]) * scale_x))
                    y1 = max(0, int(float(xyxy[1]) * scale_y))
                    x2 = min(original_w, int(float(xyxy[2]) * scale_x))
                    y2 = min(original_h, int(float(xyxy[3]) * scale_y))

                    class_idx = int(cls)
                    class_name = self.class_names[class_idx] if class_idx < len(self.class_names) else str(class_idx)

                    detections.append({
                        'class': class_name,
                        'confidence': float(conf),
                        'bbox': [x1, y1, x2, y2],
                        'model_version': self.model_version
                    })

            return {
                'status': 'success',
                'detections': detections,
                'image_size': [original_h, original_w],
                'model_info': {
                    'name': 'YOLOv7',
                    'version': self.model_version,
                    'classes': self.class_names
                }
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }
