import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import os

# Load VGG16 feature extractor
vgg16 = VGG16(weights="imagenet", include_top=False)

# Load trained classifier
classifier = tf.keras.models.load_model("../model/image_classifier.h5")

IMAGE_SIZE = (224, 224)


def predict_image(img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError("Image file not found.")

    try:
        img = image.load_img(img_path, target_size=IMAGE_SIZE)
    except:
        raise ValueError("Invalid image file.")

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    features = vgg16.predict(img_array)
    features = features.flatten().reshape(1, -1)

    prob = classifier.predict(features)[0][0]

    if prob >= 0.5:
        label = "Drug"
        confidence = prob * 100
    else:
        label = "Non-Drug"
        confidence = (1 - prob) * 100

    return label, confidence
