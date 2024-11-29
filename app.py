from flask import Flask, render_template, request, session
import random
import json
import os
import numpy as np
import faiss
import google.generativeai as genai
from flask import current_app

# Configure the Gemini API
genai.configure(api_key='YOUR_API_KEY')  # Replace with your actual API key
app = Flask(__name__)
app.secret_key = os.urandom(24)

def load_rag_system(folder_path):
    try:
        # Use faiss.read_index to load the index
        index = faiss.read_index(os.path.join(folder_path, "index.faiss"))
        
        # Load documents from JSON
        with open(os.path.join(folder_path, "documents.json"), "r") as f:
            documents = json.load(f)
        
        sorted_data = sorted(documents.values(), key=lambda x: x['doc_id'])
        documents = [item['text'] for item in sorted_data]
        return index, documents
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading RAG system: {e}")
        return None, None

def embed_text(text):
    try:
        embedding_result = genai.embed_content(
            model='models/embedding-001',
            content=text,
            task_type='retrieval_query'
        )
        return embedding_result
    except Exception as e:
        print(f"Error embedding text: {e}")
        return None

def retriever(query, index, documents, k=2):
    if not index or not documents:
        return []
    embedding_result = embed_text(query)
    if not embedding_result:
        return []

    query_embedding = np.array(embedding_result['embedding']).astype('float32').reshape(1, -1)
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, k)
    retrieved_docs = [documents[i] for i in indices[0]]
    return retrieved_docs

def generate_response(query, retrieved_docs):
    model = genai.GenerativeModel('models/gemini-1.5-flash')
    prompt = f"""
    You are a helpful assistant helping a user guess a word from a poem.
    You should only provide 2 hints without giving the answer directly.
    Use the retrieved documents below to craft creative, engaging,
    and progressively more revealing hints for the user based on their query.

    Retrieved Documents (poems): {' '.join(retrieved_docs)}

    User Query: {query}

    Instructions:
    1. Analyze the retrieved poems for context related to the query.
    2. Generate hints that help the user deduce the correct word step by step, but do not give the answer directly.
    3. If the user is close to the correct word, acknowledge their proximity and provide an encouraging hint.
    4. Ensure hints are informative yet cryptic enough to maintain the challenge.

    Your response should only contain the hints, nothing else.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Hmm, could you try another guess?"

@app.route("/", methods=["GET", "POST"])
def index():
    if "word" not in session:
        docs = ['Gold', 'Lion', 'Turtle', 'Water']  # Your word list
        session["word"] = random.choice(docs)
        session["guesses_left"] = 4
        index_path = session['word']
        
        # Load index and documents
        index, documents = load_rag_system(f'./{index_path}')
        
        current_app.config['CURRENT_INDEX'] = index
        current_app.config['CURRENT_DOCUMENTS'] = documents

        if index is None or documents is None:
            return "Error initializing game. Check file paths and configurations.", 500

    if request.method == "POST":
        guess = request.form.get("guess", "").lower()
        
        # Check if guess contains the word
        if session["word"].lower() in guess:
            message = "Well done! 🎉 You guessed the word correctly!"
            retrieved_poems = None
            hints = None
            session.clear()
        else:
            session["guesses_left"] -= 1
            
            if session["guesses_left"] > 0:
                # Retrieve poems and generate hints
                retrieved_poems = retriever(
                    guess, 
                    current_app.config['CURRENT_INDEX'], 
                    current_app.config['CURRENT_DOCUMENTS']
                )
                hints = generate_response(guess, retrieved_poems)
                
                message = f"Keep trying!"
            else:
                message = f"Game Over! 😞 The word was '{session['word']}'"
                retrieved_poems = None
                hints = None
                session.clear()

        return render_template(
            "index.html", 
            message=message, 
            guesses_left=session.get("guesses_left"),
            retrieved_poems=retrieved_poems,
            hints=hints
        )

    return render_template("index.html", message=None, guesses_left=session.get("guesses_left"))

if __name__ == "__main__":
    app.run(debug=True)