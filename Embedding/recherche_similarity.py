import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EMBEDDINGS_DIRECTORY = os.path.join(
    BASE_DIR,
    "..",
    "data",
    "dataset_embedding",
)


def prediction_pokemon(solo_image_embedding, top_k=3):
    results = []

    solo_image_embedding_reshaped = solo_image_embedding.reshape(1, -1)

    for pokemon in os.listdir(EMBEDDINGS_DIRECTORY):
        pokemon_embedding_path = os.path.join(EMBEDDINGS_DIRECTORY, pokemon)

        for file in os.listdir(pokemon_embedding_path):
            if file.endswith(".npy"):
                embedding_path = os.path.join(pokemon_embedding_path, file)
                embedding = np.load(embedding_path).reshape(1, -1)

                similarity = cosine_similarity(solo_image_embedding_reshaped, embedding)[0][0]

                results.append({
                    "pokemon": pokemon,
                    "file": file,
                    "similarity": float(similarity)
                })

    results.sort(key=lambda x: x["similarity"], reverse=True)
    resultats_topk =  results[:top_k]


    #Choix du résultat

    pokemon_scores = defaultdict(float)
    pokemon_counts = defaultdict(int)

    for res in resultats_topk:
        p = res["pokemon"]
        s = res["similarity"]

        pokemon_scores[p] += s
        pokemon_counts[p] += 1

    final_scores = {}

    for p in pokemon_scores:
        avg_similarity = pokemon_scores[p] / pokemon_counts[p]
        frequency_bonus = math.log(1 + pokemon_counts[p])
        final_scores[p] = avg_similarity * frequency_bonus

    best_pokemon = max(final_scores.items(), key=lambda x: x[1])

    return {
        "predicted_pokemon": best_pokemon[0],
        "final_score": best_pokemon[1],
        "topk_images": resultats_topk,
        "pokemon_scores": final_scores
    }                 
