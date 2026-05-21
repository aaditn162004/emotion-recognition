import os
import sys
import io
import base64

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from model import EmotionCNN

EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_model = None
_mtcnn = None
_transform = None


def load_models():
    global _model, _mtcnn, _transform

    model_path = os.path.join(os.path.dirname(__file__), '..', 'saved_models', 'emotion_model.pth')

    _model = EmotionCNN().to(device)
    _model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    _model.eval()

    _mtcnn = MTCNN(keep_all=True, device=device)

    _transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    print(f"Models loaded on {device}")


def predict(image: Image.Image) -> list:
    if _model is None:
        load_models()

    rgb = image.convert('RGB')
    boxes, _ = _mtcnn.detect(rgb)

    if boxes is None:
        return []

    w, h = rgb.size
    results = []

    for box in boxes:
        x1 = max(0, int(box[0]))
        y1 = max(0, int(box[1]))
        x2 = min(w, int(box[2]))
        y2 = min(h, int(box[3]))

        face = rgb.crop((x1, y1, x2, y2))
        if face.size[0] == 0 or face.size[1] == 0:
            continue

        tensor = _transform(face).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = _model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            idx = int(torch.argmax(probs))

        results.append({
            'box': [x1, y1, x2, y2],
            'emotion': EMOTION_LABELS[idx],
            'confidence': round(float(probs[idx]), 4),
            'all_emotions': {
                label: round(float(probs[i]), 4)
                for i, label in enumerate(EMOTION_LABELS)
            },
        })

    return results
