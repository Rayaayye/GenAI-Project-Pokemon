from Rag_Agent.llm import get_llm
from Rag_Agent.prompts import PROMPT_TOT
from Rag_Agent.retrieval import get_chroma_db
from Rag_Agent.json_retriever import load_json_pokemon

def tot_agent_run(base_pokemon, enemy_team):
    llm = get_llm()
    retriever = get_chroma_db(k=12)


    retrieval_query = f"counter team for {base_pokemon} against {', '.join(enemy_team)}"
    documents = retriever.invoke(retrieval_query)


    candidates_context = "\n".join(
        [doc.page_content for doc in documents]
    )

    base_data = load_json_pokemon(base_pokemon)
    ennemy_data = [load_json_pokemon(pokemon) for pokemon in enemy_team]


    prompt = PROMPT_TOT.format(
        BASE_POKEMON=base_data,
        ENEMY_TEAM=ennemy_data
    ) + f"""

    Candidate Pokémon Pool:
    {candidates_context}
    """


    response = llm.invoke(prompt)
    return response.content