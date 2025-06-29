from django.shortcuts import render
import tensorflow as tf
import numpy as np
from PIL import Image
import os

model = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), 'model.h5'))

labels = ['Cocci', 'Healthy', 'NCD', 'Salmo']

def predict_disease(request):
    prediction = None
    if request.method == 'POST' and request.FILES.get('image'):
        image = Image.open(request.FILES['image']).resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        result = model.predict(image_array)
        prediction = labels[np.argmax(result)]
    return render(request, 'detector/index.html', {'prediction': prediction})
