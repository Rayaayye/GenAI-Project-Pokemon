from tensorflow.keras.models import load_model, Model
import os
import streamlit as st

@st.cache_resource
def loader_dataset_finetuner():
    """Load the fine-tuned model and cache it to avoid reloading."""

    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(
        base_dir,
        "..",
        "models",
        "finetuned_efficientnetb0_pour_pokemon.h5"
    )
    
    #load the model
    model_finetune = load_model(model_path)

    # Extract the embedding layer from the full model
    model_finetune = Model(inputs=model_finetune.input, outputs=model_finetune.get_layer("embedding").output)

    return model_finetune