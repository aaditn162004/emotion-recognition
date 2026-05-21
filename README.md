# Emotion Recognition

Real-time facial emotion detection using a PyTorch CNN trained on FER2013. The browser captures your webcam, sends frames to a FastAPI backend, and displays live emotion predictions with confidence scores.

**7 emotions:** angry · disgust · fear · happy · neutral · sad · surprise

---

## Live demo

**No setup needed** — just open the link below in Chrome or Edge, allow camera access, and point at your face:

🔗 **[https://aaditn162004.github.io/emotion-recognition/](https://aaditn162004.github.io/emotion-recognition/)**

---

## Project structure

```
emotion_recognition/
├── backend/
│   ├── main.py          # FastAPI app
│   ├── inference.py     # Model loading & prediction
│   └── requirements.txt
├── docs/                    # served by GitHub Pages
│   ├── index.html
│   ├── style.css
│   └── script.js        # ← API_URL is set here
├── src/
│   ├── model.py         # CNN architecture
│   ├── train.py         # Training script
│   ├── app.py           # Original OpenCV webcam app
│   └── data_loader.py
├── saved_models/        # model stored on HF Hub (not in git)
├── Dockerfile           # HF Spaces deployment
└── .gitignore           # Excludes data/ (35 K images)
```

---

## Local development

### 1 · Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
# API runs at http://localhost:8000
```

Test it:
```bash
curl http://localhost:8000/health
```

### 2 · Frontend

Open `docs/index.html` directly in your browser — no build step needed.

> The frontend reads `API_URL = 'http://localhost:8000'` in `docs/script.js` for local dev.  
> Allow camera access when prompted.

---

## Deployment

### Backend → Hugging Face Spaces

1. Upload your model to HF Hub: `huggingface-cli upload <username>/emotion-recognition-model saved_models/emotion_model.pth emotion_model.pth --repo-type model`
2. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space) → SDK: **Docker**
3. Push the backend code (no model binary) to the Space's git repo
4. The `Dockerfile` downloads the model from HF Hub at build time
5. Your API URL: `https://<username>-emotion-recognition.hf.space`

### Frontend → GitHub Pages

1. In your GitHub repo → **Settings → Pages**.
2. Source: **Deploy from branch** → branch `main` → folder `/docs`.
3. Save. GitHub Pages URL: `https://<your-username>.github.io/<repo-name>/`

**Before pushing:** update `API_URL` in `docs/script.js`:
```js
const API_URL = 'https://<username>-emotion-recognition.hf.space';
```

> GitHub Pages requires HTTPS. Hugging Face Spaces runs on HTTPS. ✓

---

## API reference

### `GET /health`
```json
{ "status": "ok", "device": "cpu" }
```

### `POST /predict`
**Request body:**
```json
{ "image": "<base64-encoded JPEG or PNG>" }
```
**Response:**
```json
{
  "faces": [
    {
      "box": [x1, y1, x2, y2],
      "emotion": "happy",
      "confidence": 0.923,
      "all_emotions": {
        "angry": 0.01, "disgust": 0.003, "fear": 0.008,
        "happy": 0.923, "neutral": 0.04, "sad": 0.012, "surprise": 0.004
      }
    }
  ]
}
```

---

## Re-training

```bash
cd src
python train.py          # trains for 50 epochs, saves to ../saved_models/
```

Dataset: FER2013 (place in `data/fer2013/train/` and `data/fer2013/test/`).

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| Model | PyTorch CNN (EmotionCNN) |
| Face detection | MTCNN via facenet-pytorch |
| Dataset | FER2013 (28 709 train / 7 178 test images) |
| Backend | FastAPI + uvicorn |
| Frontend | Vanilla HTML / CSS / JS |
| Deployment | Hugging Face Spaces (backend) + GitHub Pages (frontend) |
