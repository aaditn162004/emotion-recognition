import base64
import io

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

import inference

app = FastAPI(title='Emotion Recognition API', version='1.0.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['GET', 'POST'],
    allow_headers=['*'],
)


class PredictRequest(BaseModel):
    image: str  # base64-encoded JPEG/PNG


@app.on_event('startup')
async def startup():
    inference.load_models()


@app.get('/health')
def health():
    return {'status': 'ok', 'device': str(inference.device)}


@app.post('/predict')
def predict(req: PredictRequest):
    try:
        img_bytes = base64.b64decode(req.image)
        image = Image.open(io.BytesIO(img_bytes))
        faces = inference.predict(image)
        return {'faces': faces}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
