from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np
import uvicorn
import tensorflow as tf

model = tf.keras.models.load_model("./saved_model/1/Model2.h5")

solutions = {
    'Black spot': 'Spray with a fungicide containing myclobutanil, trifloxystrobin, or copper',
    'Downy mildew': 'Spray with a fungicide containing mancozeb or copper',
    'Rust': 'Spray with a fungicide containing propiconazole or tebuconazole',
    'Mosaic': 'Remove infected leaves and prune infected stems. There is no cure for this virus',
    'Powdery mildew': 'Spray with a fungicide containing myclobutanil or trifloxystrobin',
    'Fresh Leaf':'No treatment is required. Keep taking care of the plant in the same way as you have been doing till now.'
}

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    image = image.resize((256, 256))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    prediction = model.predict(image_array)
    classes = ['Black Spot','Downy mildew', 'Fresh Leaf', 'Mosaic', 'Powdery mildew', 'Rust']
    predicted_class = classes[np.argmax(prediction)]
    solution = solutions[predicted_class]
    confidence = np.max(prediction[0])
    print(predicted_class, confidence)
    print(solution)
    return {
        'class': predicted_class,
        'confidence': float(confidence),
        'solution': solution
    }
if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
