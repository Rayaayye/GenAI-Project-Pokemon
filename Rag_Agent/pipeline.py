import json
from Rag_Agent.agent_react import react_agent_run
from Rag_Agent.agent_treeofthoughts import tot_agent_run
from Rag_Agent.agent_selfcorrection import self_correction_agent_run
from Rag_Agent.utils import extract_json_from_text

# Executes the multi-agent pipeline to generate an optimal Pokémon team strategy

def pipeline_run(base_pokemon, enemy_team):

    # Step 1: Tree of Thoughts agent - generates the team composition
    # The ToT agent will generate 3 possible teams and choose the one that counters the most the ennemy team

    tot_raw_response = tot_agent_run(base_pokemon, enemy_team)
    # Extract JSON data from the raw response
    tot_response = extract_json_from_text(tot_raw_response)

    # Extract the final team and combine strategy with a little justification
    final_team = tot_response["final_team"]
    initial_strategy = tot_response["strategy"] + "\nJustification: " + tot_response["justification"]

    # Step 2: ReAct agent - Analyzes our team and the ennemy team to determine the best strategy to beat the ennemy

    react_strategy = react_agent_run(
        player_pokemons=final_team,
        opponent_pokemons=enemy_team
    )

    # Step 3: Self-Correction agent - reviews the strategy and improves it

    self_correction_strategy = self_correction_agent_run(
        strategy=react_strategy,
        initial_team=final_team,
        enemy_team=enemy_team
    )

    # Return the final team, refined strategy, and all the reasoning traces
    return {
        "team": final_team,
        "strategy": self_correction_strategy["final_strategy"],
        "reasoning_trace": {
            "tot_raw": tot_raw_response,
            "tot_parsed": tot_response,
            "tot_initial_strategy": initial_strategy,   
            "react": react_strategy,
            "self_correction": self_correction_strategy
        }
    }