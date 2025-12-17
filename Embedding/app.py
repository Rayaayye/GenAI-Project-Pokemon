from fonction_embedding_image_solo import one_image_embedding
from recherche_similarity import prediction_pokemon

import streamlit as st
import tempfile




st.set_page_config(page_title="Pokémon Search", layout="centered")

st.title("Pokémon Image Similarity Search")
st.write("Upload an image of a Pokémon to find the most similar one")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', width=300)

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    with st.spinner('Processing...'):
        embedding_image = one_image_embedding(temp_file_path)
        result = prediction_pokemon(embedding_image, top_k=3)

    st.success('Done!')

    # ---------- Pokémon final ----------
    st.subheader("✅ Predicted Pokémon")
    st.write(f"**{result['predicted_pokemon']}**")
    st.write(f"Final score: **{result['final_score']:.4f}**")

    # ---------- Top-5 images ----------
    st.subheader("🔍 Top 4 Most Similar Images")

    for rank, res in enumerate(result["topk_images"], start=1):
        st.write(
            f"**{rank}. {res['pokemon']}** "
            f"(image: {res['file']}) — "
            f"similarity: **{res['similarity']:.4f}**"
        )


