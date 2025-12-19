import json
import os
import re


# We had bugs with paths before so we did that to not have any problems when running the project

#Define paths

BASE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

POKEDEX_DIRECTORY = os.path.join(
    BASE_DIRECTORY,
    "..",
    "data",
    "pokedex"
)

# Normalizes Pokémon names 
# If we have Pikachu, PIKACHU or pikachu, it will be the same for the program
# If we have Mime Jr. or mime-jr, the program will know its the same
def normalize(name: str) -> str:
    # Convert to lowercase
    name = name.lower()
    # Remove all numbers from the name
    name = re.sub(r"\d+", "", name)
    # Remove separators: spaces, hyphens, and underscores
    name = re.sub(r"[\s\-_]", "", name)
    return name

# Function that loads Pokémon JSON data by name
def load_json_pokemon(pokemon_name: str) -> dict:
    # Normalize the Pokémon name=
    target = normalize(pokemon_name)

    # Iterate through the files of the Pokédex directory
    for filename in os.listdir(POKEDEX_DIRECTORY):
        # Skip files that are not JSON
        if not filename.endswith(".json"):
            continue

        # Remove the .json extension from the filename
        filename_no_ext = filename.replace(".json", "")
        # Normalize the filename for comparison
        normalized_filename = normalize(filename_no_ext)

        # Check if the normalized filename matches the target Pokémon
        if normalized_filename == target:
            # Open and load the JSON file
            with open(
                os.path.join(POKEDEX_DIRECTORY, filename),
                "r",
                encoding="utf-8"
            ) as f:
                # Return the parsed JSON data
                return json.load(f)

    # Raise an error if the Pokémon is not found in the directory
    raise ValueError(f"Pokémon '{pokemon_name}' not found in the Pokedex directory.")
