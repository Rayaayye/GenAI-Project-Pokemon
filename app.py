from Embedding.fonction_embedding_image_solo import one_image_embedding
from Embedding.recherche_similarity import prediction_pokemon

import streamlit as st
import tempfile
import os
import json


from dotenv import load_dotenv
load_dotenv()

from Rag_Agent.pipeline import pipeline_run


BASE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

POKEDEX_DIRECTORY = os.path.join(
    BASE_DIRECTORY,
    "data",
    "pokedex"
)

def get_all_pokemon_names():
    return sorted([
        f.replace(".json", "")
        for f in os.listdir(POKEDEX_DIRECTORY)
        if f.endswith(".json")
    ])


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

    predicted_pokemon = result["predicted_pokemon"]

    st.success("Pokémon identified!")

    st.subheader("✅ Predicted Pokémon")
    st.write(f"**{predicted_pokemon}**")
    st.write(f"Confidence score: **{result['final_score']:.4f}**")

    st.subheader("🔍 Top Similar Images")
    for rank, res in enumerate(result["topk_images"], start=1):
        st.write(
            f"**{rank}. {res['pokemon']}** "
            f"(image: {res['file']}) — "
            f"similarity: **{res['similarity']:.4f}**"
        )

    # Step 2: Enemy Team Selection

    st.divider()
    st.subheader("⚔️ Select Enemy Team")

    all_pokemon = get_all_pokemon_names()

    enemy_team = st.multiselect(
        "Choose exactly 3 enemy Pokémon",
        options=all_pokemon,
        max_selections=3
    )

    # Step 3 — Run Full Pipeline

    if len(enemy_team) == 3:
        if st.button("🚀 Compute Best Team & Strategy"):
            with st.spinner("🧠 Running reasoning agents (ToT → ReAct → Self-Correction)..."):
                pipeline_result = pipeline_run(
                    base_pokemon=predicted_pokemon,
                    enemy_team=enemy_team
                )

            st.success("Strategy ready!")

            # Results

            st.subheader("🏆 Recommended Team")
            st.write(pipeline_result["team"])

            st.subheader("📋 Battle Strategy")
            st.write(pipeline_result["strategy"])

            #Reasoning Trace
            with st.expander("🧩 Show reasoning trace (advanced)"):
                st.json(pipeline_result["reasoning_trace"])