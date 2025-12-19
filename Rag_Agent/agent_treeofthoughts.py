from Rag_Agent.llm import get_llm
from Rag_Agent.prompts import PROMPT_TOT
from Rag_Agent.retrieval import get_chroma_db
from Rag_Agent.json_retriever import load_json_pokemon

# Executes the Tree of Thoughts agent to find the optimal Pokémon team to counter the ennemy team
def tot_agent_run(base_pokemon, enemy_team):
    # Initialize the LLM
    llm = get_llm()
    # Initialize the vector database retriever with top-12 results
    retriever = get_chroma_db(k=12)

    # Create a search query to find 2 pokemons to complete the roaster with our base pokemon. Roaster that will be made to counter the enemy team
    retrieval_query = f"counter team for {base_pokemon} against {', '.join(enemy_team)}"
    # Retrieve candidate Pokémon documents from the vector database
    documents = retriever.invoke(retrieval_query)

    # Extract and join the content from all retrieved documents
    candidates_context = "\n".join(
        [doc.page_content for doc in documents]
    )

    # Load JSON data for the base Pokémon we uploaded
    base_data = load_json_pokemon(base_pokemon)
    # Load JSON data for each enemy Pokémon
    ennemy_data = [load_json_pokemon(pokemon) for pokemon in enemy_team]

    # Fill the Tree of Thoughts prompt template with Pokémon data
    prompt = PROMPT_TOT.format(
        BASE_POKEMON=base_data,
        ENEMY_TEAM=ennemy_data
    ) + f"""

    Candidate Pokémon Pool:
    {candidates_context}
    """

    # Send the prompt to the LLM and extract the response text
    response = llm.invoke(prompt)
    # Return the optimal team composition and reasoning
    return response.content