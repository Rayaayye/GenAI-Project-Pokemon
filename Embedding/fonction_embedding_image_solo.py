import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing import image_dataset_from_directory
from Embedding.loading_model import loader_dataset_finetuner

import numpy as np

import os
import numpy as np
from tqdm import tqdm

# Image size we need for EfficientNetB0 model input
IMAGE_SIZE = (224, 224)

# Function that generate a normalized embedding vector for a single Pokémon image
def one_image_embedding(image_path):

    # Load the fine-tuned EfficientNetB0 model
    model_finetune = loader_dataset_finetuner()
    
    # Load the image and resize it to (224, 224)
    img = load_img(image_path, target_size=IMAGE_SIZE)
    # Convert the image to a numpy array
    img_array = img_to_array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Apply EfficientNet preprocessing normalization
    img_array = preprocess_input(img_array)

    # Generate the embedding vector for the image
    embedding = model_finetune.predict(img_array, verbose=0)[0]
    # We apply L2 normalization
    embedding = embedding / np.linalg.norm(embedding)

    return embedding