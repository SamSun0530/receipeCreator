import os
from openai import OpenAI
from pinecone import Pinecone

# 1. Initialize Clients
oa_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "fridge-ai-recipes"
index = pc.Index(index_name)

def get_matching_recipes(ingredients: list):
    # Convert list of ingredients to a string for embedding
    query_text = ", ".join(ingredients)
    
    # 2. Generate Embedding (must match the model used for your dataset)
    response = oa_client.embeddings.create(
        input=[query_text],
        model="text-embedding-3-small"
    )
    query_vector = response.data[0].embedding
    
    # 3. Query Pinecone
    search_results = index.query(
        vector=query_vector,
        top_k=3,
        include_metadata=True
    )
    
    # Format the results cleanly
    recipes = []
    for match in search_results['matches']:
        recipes.append({
            "Recipe": match['metadata']['name'],
            "score": round(match['score'], 3),
            "ingredients": match['metadata']['ingredients'],
            "instructions": match['metadata']['steps'][:150] + "..."
        })

    print("!!!!!")
    print(recipes)
    print("!!!!!")
    
    return recipes