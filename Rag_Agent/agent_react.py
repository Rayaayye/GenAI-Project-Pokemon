from Rag_Agent.llm import get_llm
from Rag_Agent.prompts import PROMPT_REACT
from Rag_Agent.json_retriever import load_json_pokemon

#Function to execute the ReAct agent to analyze the Pokémon battle scenario and create a strategy to win
def react_agent_run(player_pokemons, opponent_pokemons):

    # Initialize the LLM
    llm = get_llm()

    # Load the JSON data for each player Pokémon
    player_pokemons_data = [load_json_pokemon(pokemon) for pokemon in player_pokemons]
    # Load the JSON data for each opponent Pokémon
    opponent_pokemons_data = [load_json_pokemon(pokemon) for pokemon in opponent_pokemons]  

    # Fill the prompt template with player and opponent Pokémon data
    prompt = PROMPT_REACT.format(
        PLAYER_POKEMONS = player_pokemons_data,
        OPPONENT_POKEMONS = opponent_pokemons_data,
    )

    # Send the prompt to the LLM and extract the response text
    response = llm.invoke(prompt).content
    # Return the battle strategy and recommendations
    return response