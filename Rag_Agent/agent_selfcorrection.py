from Rag_Agent.llm import get_llm
from Rag_Agent.prompts import PROMPT_SELFCORRECTION
from Rag_Agent.json_retriever import load_json_pokemon

# Executes the self-correction agent to review and improve a Pokémon battle strategy
def self_correction_agent_run(strategy, initial_team, enemy_team):
    # Initialize the LLM
    llm = get_llm()

    # Load JSON data for each player Pokémon
    initial_team_data = [load_json_pokemon(pokemon) for pokemon in initial_team]
    # Load JSON data for each enemy Pokémon
    enemy_team_data = [load_json_pokemon(pokemon) for pokemon in enemy_team]

    # Fill the prompt template with player and opponent Pokémon data
    prompt = PROMPT_SELFCORRECTION.format(
        PLAYER_POKEMONS = initial_team_data,
        OPPONENT_POKEMONS = enemy_team_data,
    )

    # Append the previous strategy to the prompt
    prompt += "\n\nPrevious Strategy to Review:\n"
    # Add for the evaluation
    prompt += strategy

    # Send the prompt to the LLM and extract the response text
    response = llm.invoke(prompt).content
    # Return the improved strategy and corrections
    return response