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

IMAGE_SIZE = (224, 224)

def one_image_embedding(image_path):

    model_finetune = loader_dataset_finetuner()
    
    # Load and preprocess the image
    img = load_img(image_path, target_size=IMAGE_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Get the embedding
    embedding = model_finetune.predict(img_array, verbose=0)[0]
    #L2 Normalization
    embedding = embedding / np.linalg.norm(embedding)

    return embedding