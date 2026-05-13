import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

# ------------------------------------------------
# Rebuild model architecture manually to avoid
# Keras version mismatch when loading .h5 config.
# Architecture extracted from the saved model config:
#   Input(25088) -> Dense(512,relu) -> BN -> Drop(0.5)
#   -> Dense(256,relu) -> BN -> Drop(0.4)
#   -> Dense(128,relu) -> BN -> Drop(0.3)
#   -> Dense(1,sigmoid)
# ------------------------------------------------


def _build_classifier():
    """Reconstruct the classifier architecture and load saved weights."""
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(25088,)),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    h5_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "model", "image_classifier.h5"
    )

    # Build the model first so weight shapes are initialized
    model.build((None, 25088))

    # Load only the weights from the .h5 file (skip config)
    model.load_weights(h5_path)

    return model


# ------------------------------------------------
# LOAD MODELS (loaded once at import time)
# ------------------------------------------------

vgg16 = VGG16(weights="imagenet", include_top=False)
classifier = _build_classifier()

IMAGE_SIZE = (224, 224)


def predict_image(img_path):
    """
    Predict whether an image is Drug or Non-Drug.
    Returns: (label, confidence_percentage)
    """
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")

    try:
        img = image.load_img(img_path, target_size=IMAGE_SIZE)
    except Exception:
        raise ValueError(f"Invalid or corrupted image file: {img_path}")

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    features = vgg16.predict(img_array, verbose=0)
    features = features.flatten().reshape(1, -1)

    prob = classifier.predict(features, verbose=0)[0][0]

    if prob >= 0.48:
        label = "Drug"
        confidence = prob * 100
    else:
        label = "Non-Drug"
        confidence = (1 - prob) * 100

    return label, confidence
