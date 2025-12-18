import json
import os
import re


BASE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))


POKEDEX_DIRECTORY = os.path.join(
    BASE_DIRECTORY,
    "..",
    "data",
    "pokedex"
)

def normalize(name: str) -> str:
    name = name.lower()
    name = re.sub(r"\d+", "", name)          # remove numbers
    name = re.sub(r"[\s\-_]", "", name)      # remove separators
    return name

def load_json_pokemon(pokemon_name: str) -> dict:
    target = normalize(pokemon_name)

    for filename in os.listdir(POKEDEX_DIRECTORY):
        if not filename.endswith(".json"):
            continue

        filename_no_ext = filename.replace(".json", "")
        normalized_filename = normalize(filename_no_ext)

        if normalized_filename == target:
            with open(
                os.path.join(POKEDEX_DIRECTORY, filename),
                "r",
                encoding="utf-8"
            ) as f:
                return json.load(f)

    raise ValueError(f"Pokémon '{pokemon_name}' not found in the Pokedex directory.")
