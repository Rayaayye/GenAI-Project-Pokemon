from Rag_Agent.llm import get_llm
from Rag_Agent.prompts import PROMPT_SELFCORRECTION
from Rag_Agent.json_retriever import load_json_pokemon

def self_correction_agent_run(strategy, initial_team, enemy_team):
    llm = get_llm()

    initial_team_data = [load_json_pokemon(pokemon) for pokemon in initial_team]
    enemy_team_data = [load_json_pokemon(pokemon) for pokemon in enemy_team]

    prompt = PROMPT_SELFCORRECTION.format(
        PLAYER_POKEMONS = initial_team_data,
        OPPONENT_POKEMONS = enemy_team_data,
    )

    prompt += "\n\nPrevious Strategy to Review:\n"
    prompt += strategy

    response = llm.invoke(prompt).content
    return response