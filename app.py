from Embedding.fonction_embedding_image_solo import one_image_embedding
from Embedding.recherche_similarity import prediction_pokemon

import streamlit as st
import tempfile
import os
import json


from dotenv import load_dotenv
load_dotenv()

from Rag_Agent.pipeline import pipeline_run

# We had bugs with paths before so we did that to not have any problems when running the project

#Define paths

BASE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

POKEDEX_DIRECTORY = os.path.join(
    BASE_DIRECTORY,
    "data",
    "pokedex"
)

# Retrieves and returns a sorted list of all Pokémon names from JSON files

def get_all_pokemon_names():
    """Get all Pokemon names from the pokedex directory, formatted with capitals."""
    return sorted([
        f.replace(".json", "").replace("-", " ").title()
        for f in os.listdir(POKEDEX_DIRECTORY)
        if f.endswith(".json")
    ])

def get_pokemon_image_path(pokemon_name):
    """Get the first image path for a given Pokemon from the dataset."""
    # Remove number prefix if present (e.g., "119 Seaking" -> "Seaking")
    # The format uses underscore: "119_Seaking"
    if "_" in pokemon_name:
        pokemon_name = pokemon_name.split("_", 1)[1]
    
    # Convert formatted name back to folder name (lowercase, with hyphens)
    folder_name = pokemon_name.lower().replace(" ", "-")
    
    dataset_path = os.path.join(BASE_DIRECTORY, "data", "pokemon-dataset-1000", folder_name)
    
    if os.path.exists(dataset_path):
        # Get the first image in the folder
        for file in os.listdir(dataset_path):
            if file.endswith(('.jpg', '.jpeg', '.png')):
                return os.path.join(dataset_path, file)
    
    return None


# Streamlit
st.set_page_config(page_title="Pokémon Battle Advisor", layout="centered")

st.title("🧠 Pokémon Battle Advisor")
st.write("Upload a Pokémon image, select an enemy team, and get a full battle strategy.")

uploaded_file = st.file_uploader(
    "📷 Upload a Pokémon image",
    type=["jpg", "jpeg", "png"]
)

# Step 1: Image Recognition
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", width=300)

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    with st.spinner("🔍 Identifying Pokémon..."):
        embedding_image = one_image_embedding(temp_file_path)
        result = prediction_pokemon(embedding_image, top_k=3)

    # Format the predicted Pokemon name with capitals
    predicted_pokemon = result["predicted_pokemon"].replace("-", " ").title()

    st.success("Pokémon identified!")

    st.subheader("✅ Predicted Pokémon")
    st.write(f"**{predicted_pokemon}**")
    st.write(f"Confidence score: **{result['final_score']:.2f}%**")

    st.subheader("🔍 Top 3 Similar Images")
    
    # Display top 3 similar images in columns
    cols = st.columns(3)
    for idx, res in enumerate(result["topk_images"]):
        with cols[idx]:
            # Get the specific image that matches the embedding file
            pokemon_folder = res['pokemon']
            embedding_filename = res['file']  # e.g., 'pikachu_5.npy'
            
            # Extract the base name and try different extensions
            base_name = embedding_filename.replace('.npy', '')
            
            # Try to find the matching image in pokemon-dataset-1000
            image_path = None
            dataset_folder = os.path.join(BASE_DIRECTORY, "data", "pokemon-dataset-1000", pokemon_folder)
            
            if os.path.exists(dataset_folder):
                # Try common image extensions
                for ext in ['.png', '.jpg', '.jpeg']:
                    potential_path = os.path.join(dataset_folder, base_name + ext)
                    if os.path.exists(potential_path):
                        image_path = potential_path
                        break
            
            # Display image if found
            if image_path and os.path.exists(image_path):
                st.image(image_path, use_container_width=True)
            else:
                st.write("🖼️ Image not found")
            
            st.write(f"**{res['pokemon'].replace('-', ' ').title()}**")
            st.write(f"Similarity: {res['similarity']*100:.2f}%")

    # Step 2: Enemy Team Selection

    st.divider()
    st.subheader("⚔️ Select Enemy Team")

    all_pokemon = get_all_pokemon_names()

    enemy_team = st.multiselect(
        "Choose exactly 3 enemy Pokémon",
        options=all_pokemon,
        max_selections=3
    )
    
    # Display selected enemy team with images
    if len(enemy_team) > 0:
        st.write("**Selected Enemy Team:**")
        enemy_cols = st.columns(len(enemy_team))
        for idx, pokemon in enumerate(enemy_team):
            with enemy_cols[idx]:
                img_path = get_pokemon_image_path(pokemon)
                if img_path and os.path.exists(img_path):
                    st.image(img_path, use_container_width=True)
                else:
                    # Show placeholder if image not found
                    st.write("🖼️ Image not found")
                st.write(f"**{pokemon}**")

    # Step 3 — Run Full Pipeline

    if len(enemy_team) == 3:
        if st.button("🚀 Compute Best Team & Strategy"):
            with st.spinner("🧠 Running reasoning agents (ToT → ReAct → Self-Correction)..."):
                # Convert enemy team back to lowercase format for pipeline
                enemy_team_lowercase = [name.lower().replace(" ", "-") for name in enemy_team]
                
                pipeline_result = pipeline_run(
                    base_pokemon=predicted_pokemon.lower().replace(" ", "-"),
                    enemy_team=enemy_team_lowercase
                )

            st.success("Strategy ready!")

            # Results

            st.subheader("🏆 Recommended Team")
            
            # Display the team with images
            team_pokemon = pipeline_result["team"]
            if isinstance(team_pokemon, list):
                team_cols = st.columns(len(team_pokemon))
                for idx, pokemon in enumerate(team_pokemon):
                    with team_cols[idx]:
                        # Format pokemon name for display
                        pokemon_display = pokemon.replace("-", " ").title()
                        img_path = get_pokemon_image_path(pokemon_display)
                        
                        if img_path and os.path.exists(img_path):
                            st.image(img_path, use_container_width=True)
                        st.write(f"**{pokemon_display}**")
            else:
                st.write(team_pokemon)

            st.subheader("📋 Battle Strategy")
            st.write(pipeline_result["strategy"])

            #Reasoning Trace
            with st.expander("🧩 Show reasoning trace (advanced)"):
                st.json(pipeline_result["reasoning_trace"])