import os
import time
import json
import openai
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_embeddings(courses, vector_store_path='vector_store'):
    # Get API key directly from environment variable
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    embeddings = OpenAIEmbeddings(api_key=api_key)
    texts = [course['description'] for course in courses if course['description']]
    
    if not texts:
        print("No valid descriptions found for embedding.")
        return None

    max_retries = 5
    for attempt in range(max_retries):
        try:
            embedding_vectors = embeddings.embed_documents(texts)
            vector_store = FAISS.from_embeddings(embedding_vectors)
            
            # Save the vector store to a file
            vector_store.save(vector_store_path)
            print(f"Vector store saved to {vector_store_path}.")
            return vector_store
        except openai.OpenAIError as e:
            wait_time = 2 ** attempt  # Exponential backoff
            print(f"OpenAI error: {e}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"An unexpected error occurred: {e}. Retrying in {2 ** attempt} seconds...")
            time.sleep(2 ** attempt)  # Exponential backoff
    
    print("Max retries reached. Failed to create embeddings.")
    return None

def load_courses_from_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

if __name__ == "__main__":
    courses = load_courses_from_json('courses.json')
    create_embeddings(courses)
