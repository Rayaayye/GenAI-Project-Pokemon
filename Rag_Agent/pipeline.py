import json
from Rag_Agent.agent_react import react_agent_run
from Rag_Agent.agent_treeofthoughts import tot_agent_run
from Rag_Agent.agent_selfcorrection import self_correction_agent_run
from Rag_Agent.utils import extract_json_from_text

def pipeline_run(base_pokemon, enemy_team):

    # 1. ToT

    tot_raw_response = tot_agent_run(base_pokemon, enemy_team)
    tot_response = extract_json_from_text(tot_raw_response)

    final_team = tot_response["final_team"]
    initial_strategy = tot_response["strategy"] + "\nJustification: " + tot_response["justification"]

    # 2. ReAct

    react_strategy = react_agent_run(
        player_pokemons=final_team,
        opponent_pokemons=enemy_team
    )

    # 3. Self-Correction

    self_correction_strategy = self_correction_agent_run(
        strategy=react_strategy,
        initial_team=final_team,
        enemy_team=enemy_team
    )

    return {
        "team": final_team,
        "strategy": self_correction_strategy,
        "reasoning_trace": {
            "tot_raw": tot_raw_response,
            "tot_parsed": tot_response,
            "tot_initial_strategy": initial_strategy,   
            "react": react_strategy,
            "self_correction": self_correction_strategy
        }
    }