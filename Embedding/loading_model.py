from tensorflow.keras.models import load_model, Model
import os

def loader_dataset_finetuner():

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(
        base_dir,
        "..",
        "models",
        "finetuned_efficientnetb0_pour_pokemon.h5"
    )
    
    model_finetune = load_model(model_path)

    model_finetune = Model(inputs=model_finetune.input, outputs=model_finetune.get_layer("embedding").output)

    return model_finetune