import os
import logging
import tensorflow as tf
import absl.logging

# Disable most TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'            # 3 = ERROR only
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'           # Turn off oneDNN logs

# Set Python logging levels
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)
absl.logging.set_verbosity(absl.logging.ERROR)



from tensorflow.keras.models import load_model
import numpy as np
import cv2
import io
from PIL import Image
from fastapi import FastAPI, UploadFile, File



model = load_model("asl_mobilenet_model.h5")

labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
          'DELETE', 'NOTHING', 'SPACE']

def preprocess_image(file, img_size=224):
    image = Image.open(io.BytesIO(file)).convert("RGB")
    image = image.resize((img_size, img_size))
    image = np.array(image).astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    return image

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "ASL Prediction API is running"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = preprocess_image(contents)

    preds = model.predict(image)
    confidence = float(np.max(preds))
    class_idx = int(np.argmax(preds))
    label = labels[class_idx]

    return {
        "prediction": label,
        "confidence": round(confidence, 3)
    }