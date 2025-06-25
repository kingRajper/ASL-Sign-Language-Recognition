# 🤟 ASL Sign Language Recognition API

A deep learning-powered FastAPI application for recognizing American Sign Language (ASL) hand gestures in real time from images. It uses a fine-tuned MobileNetV2 model trained on a dataset of 87,000 ASL gesture images across 29 classes (A–Z, SPACE, DELETE, NOTHING).

---

## 📂 Dataset

- **Source**: [Kaggle - ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- **Size**: 87,000 training images, 29 classes (`A-Z`, `SPACE`, `DELETE`, `NOTHING`)
- **Image Size**: 200x200 pixels (resized to 224x224 for MobileNetV2)

---

## 🚀 Features

- ✅ FastAPI-powered web server with RESTful prediction endpoint
- ✅ Trained with MobileNetV2 + Dropout + BatchNorm
- ✅ Real-time inference support (via file upload)
- ✅ Real-time webcam-based prediction using OpenCV
- ✅ Returns predicted class and confidence
- ✅ Production-ready architecture
- ✅ Silent TensorFlow logs for clean output

---

## 🧠 Model

- 📚 **Architecture**: MobileNetV2 (pretrained, fine-tuned)
- 🎯 **Accuracy**: ~96% train accuracy, ~84% validation accuracy
- 💾 Saved as: `asl_mobilenet_model.h5`

---

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/asl-sign-predictor-api.git
cd asl-sign-predictor-api

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
````

---

## 📦 Requirements

```
fastapi
uvicorn
tensorflow
opencv-python
numpy
python-multipart
Pillow
```

---

## 🚦 Run the API

```bash
uvicorn main:app --reload
```

> Access Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🧪 API Usage

### 🔍 Endpoint: `POST /predict/`

#### ✔️ Parameters:

* `file`: image file (JPEG, PNG)

#### ✔️ Response:

```json
{
  "prediction": "G",
  "confidence": 0.987
}
```

#### ✔️ Example using `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/predict/" \
     -F "file=@sample_asl.jpg"
```

---

## 🎥 Real-Time Prediction

You can use your webcam or camera feed to predict ASL gestures live.

Run the script (e.g., `realtime.py`) with OpenCV to:

* Access webcam
* Capture frames
* Predict ASL class in real time
* Display prediction with confidence overlay

```bash
python realtime.py
```

> Make sure your model file `asl_mobilenet_model.h5` is available in the project root or updated in the script path.

---

## 🖼️ Example Image Classes

| Class   | Meaning             |
| ------- | ------------------- |
| A–Z     | Alphabets           |
| SPACE   | Spacebar gesture    |
| DELETE  | Delete/backspace    |
| NOTHING | No gesture detected |

---

## 📁 Project Structure

```
asl_api/
├── main.py               # FastAPI app
├── realtime.py           # Webcam-based real-time prediction
├── requirements.txt
├── asl_mobilenet_model.h5
└── README.md
```

---

## 📌 To Do / Next Steps

* [ ] Add frontend (Streamlit or React)
* [ ] Deploy via Docker + Railway / Render / GCP
* [ ] Convert to TensorFlow Lite or ONNX

---

## 🧑‍💻 Author

**Iqrar Ali**
  Passionate ML Engineer
📍 Pakistan
📧 [iqrarrajper22@gmail.com](mailto:iqrarrajper22@gmail.com)
🔗 [LinkedIn](https://www.linkedin.com/in/iqrar-ali-r-9a88a3214/)
---


