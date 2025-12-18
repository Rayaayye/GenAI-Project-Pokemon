from Rag_Agent.llm import get_llm
from Rag_Agent.prompts import PROMPT_REACT
from Rag_Agent.json_retriever import load_json_pokemon

def react_agent_run(player_pokemons, opponent_pokemons):
    llm = get_llm()

    player_pokemons_data = [load_json_pokemon(pokemon) for pokemon in player_pokemons]
    opponent_pokemons_data = [load_json_pokemon(pokemon) for pokemon in opponent_pokemons]  

    prompt = PROMPT_REACT.format(
        PLAYER_POKEMONS = player_pokemons_data,
        OPPONENT_POKEMONS = opponent_pokemons_data,
    )

    response = llm.invoke(prompt).content
    return response