# src/app.py

import cv2
import torch
from facenet_pytorch import MTCNN # <-- IMPORTANT: New import
from model import EmotionCNN
from torchvision import transforms

def run_app():
    """Runs the real-time emotion recognition application."""

    # --- 1. Setup ---
    MODEL_PATH = '../saved_models/emotion_model.pth'
    EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 2. Load Model and Face Detector ---
    print("Loading model...")
    model = EmotionCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    print("Initializing PyTorch MTCNN face detector...")
    # Initialize the PyTorch-based MTCNN face detector
    mtcnn = MTCNN(keep_all=True, device=device)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # --- 3. Start Video Capture ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting webcam feed... Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- 4. Detect Faces ---
        # The new detector returns boxes and probabilities
        boxes, _ = mtcnn.detect(frame)
        
        # Check if any faces were detected
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(coord) for coord in box]
                w, h = x2 - x1, y2 - y1
                
                face_roi = frame[y1:y2, x1:x2]

                if face_roi.size == 0:
                    continue

                # --- 5. Preprocess Face ROI ---
                face_tensor = transform(face_roi).unsqueeze(0).to(device)

                # --- 6. Predict Emotion ---
                with torch.no_grad():
                    output = model(face_tensor)
                    _, predicted_idx = torch.max(output, 1)
                    emotion = EMOTION_LABELS[predicted_idx.item()]

                # --- 7. Display Result ---
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, emotion, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Emotion Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- 8. Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")

if __name__ == '__main__':
    run_app()