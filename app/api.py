from flask import Flask, request, jsonify, send_from_directory
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

app = Flask(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'))

# Load vector store
vector_store_path = 'vector_store'  # Make sure this path is correct
vector_store = None

if os.path.exists(vector_store_path):
    try:
        vector_store = FAISS.load(vector_store_path)
    except Exception as e:
        print(f"Failed to load vector store: {e}")
else:
    print("Vector store file does not exist. Rebuilding...")

@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.json
        user_input = data.get('query', '')

        if not user_input:
            return jsonify({'results': 'No input provided'})

        # Generate embedding for user input
        user_embedding = embeddings.embed_query(user_input)
        
        # Perform similarity search
        results = vector_store.search(user_embedding)
        
        return jsonify({'results': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
