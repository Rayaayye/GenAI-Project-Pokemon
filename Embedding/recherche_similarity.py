import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import math


# We had bugs with paths before so we did that to not have any problems when running the project

#Define paths

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EMBEDDINGS_DIRECTORY = os.path.join(
    BASE_DIR,
    "..",
    "data",
    "dataset_embedding",
)


# Function that will find who is the pokemon we uploaded (top 3 similarity here)
def prediction_pokemon(solo_image_embedding, top_k=3):
    # List to store all similarity results
    results = []

    # Reshape the input embedding to 2D format for cosine similarity 
    solo_image_embedding_reshaped = solo_image_embedding.reshape(1, -1)

    # Iterate through all Pokemon directories in the embeddings dataset
    for pokemon in os.listdir(EMBEDDINGS_DIRECTORY):
        pokemon_embedding_path = os.path.join(EMBEDDINGS_DIRECTORY, pokemon)

        # Load all pre-computed embedding files for the current Pokemon
        for file in os.listdir(pokemon_embedding_path):
            if file.endswith(".npy"):
                embedding_path = os.path.join(pokemon_embedding_path, file)
                # Load the pre-computed embedding
                embedding = np.load(embedding_path).reshape(1, -1)

                # Calculate cosine similarity between the input and stored embedding
                similarity = cosine_similarity(solo_image_embedding_reshaped, embedding)[0][0]

                # Store the result with Pokemon name, file, and similarity score
                results.append({
                    "pokemon": pokemon,
                    "file": file,
                    "similarity": float(similarity)
                })

    # Sort all results by similarity score from the best to the worst
    results.sort(key=lambda x: x["similarity"], reverse=True)
    # Extract the top-k most similar pokemon, in our case  top 3
    resultats_topk = results[:top_k]

    # Here when we have chosen only the pokemon that has the best similarity, sometimes there were issues, we didn't have the good one.
    #Therefore, we compute the 3 best pokemons and we get our result by the score they got and the number of time a pokemon from the same species appears in the top 3.

    # Aggregate scores by Pokemon species from top-k results
    pokemon_scores = defaultdict(float)
    pokemon_counts = defaultdict(int)

    # Sum similarity scores and count occurrences for each Pokemon
    for res in resultats_topk:
        p = res["pokemon"]
        s = res["similarity"]

        pokemon_scores[p] += s
        pokemon_counts[p] += 1

    # Calculate final weighted scores for each Pokemon
    final_scores = {}

    # Compute average similarity weighted by frequency bonus
    for p in pokemon_scores:
        avg_similarity = pokemon_scores[p] / pokemon_counts[p]
        # Apply logarithmic frequency bonus to favor Pokemon with multiple matches
        frequency_bonus = math.log(1 + pokemon_counts[p])
        final_scores[p] = avg_similarity * frequency_bonus

    # Find the Pokemon with the highest final score
    best_pokemon = max(final_scores.items(), key=lambda x: x[1])
    
    # Calculate confidence as percentage (0-100%)
    # Use the average similarity of the best pokemon
    confidence_percentage = (pokemon_scores[best_pokemon[0]] / pokemon_counts[best_pokemon[0]]) * 100

    # Return the predicted Pokemon and detailed scoring information
    return {
        "predicted_pokemon": best_pokemon[0],
        "final_score": confidence_percentage,
        "topk_images": resultats_topk,
        "pokemon_scores": final_scores
    }                 
