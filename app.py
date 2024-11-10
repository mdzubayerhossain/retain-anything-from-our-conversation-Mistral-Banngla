from flask import Flask, render_template, request, jsonify, session
import time
import numpy as np
import faiss
from dotenv import load_dotenv
from mistralai import Mistral
from langchain_community.document_loaders import TextLoader
from flask_session import Session  # Import for extended session management

# Load environment variables
load_dotenv()
api_key = "TYour Mistral API"
client = Mistral(api_key=api_key)

# Initialize the Flask app and session
app = Flask(__name__)
app.secret_key = 'Your secret key'  # Set a secret key for session management
app.config['SESSION_TYPE'] = 'filesystem'  # Store session data in a local file
Session(app)  # Initialize the session

# Load data
loader = TextLoader(r"Your Docs", encoding="utf-8")
docs = loader.load()
text = docs[0].page_content

# Chunk text data
chunk_size = 6500
chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Function to get text embedding
def get_text_embedding(input_text):
    embeddings_batch_response = client.embeddings.create(
        model="mistral-embed",
        inputs=input_text
    )
    return embeddings_batch_response.data[0].embedding

# Add a delay between API calls to avoid rate limiting
delay_seconds = 2
text_embeddings = []
for chunk in chunks:
    embedding = get_text_embedding(chunk)
    text_embeddings.append(embedding)
    time.sleep(delay_seconds)

# Convert embeddings to a NumPy array and index with Faiss
text_embeddings = np.array(text_embeddings)
d = text_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(text_embeddings)

# Function to generate a response with Mistral, using conversation history
def run_mistral(prompt, model="open-mistral-nemo"):
    messages = [{"role": "user", "content": prompt}]
    time.sleep(delay_seconds)
    chat_response = client.chat.complete(model=model, messages=messages)
    return chat_response.choices[0].message.content

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get', methods=['GET'])
def get_bot_response():
    question = request.args.get('msg')
    
    # Initialize session conversation history if it doesn't exist
    if 'conversation_history' not in session:
        session['conversation_history'] = []

    # Get embedding for the question
    question_embedding = np.array([get_text_embedding(question)])
    
    # Find the closest matching chunk
    D, I = index.search(question_embedding, k=2)
    retrieved_chunk = [chunks[i] for i in I[0]]

    # Add previous conversation history to the prompt
    history = "\n".join(session['conversation_history'])
    prompt = f"""
    নীচে প্রাসঙ্গিক তথ্য দেওয়া আছে।
    ---------------------
    {retrieved_chunk}
    ---------------------
    আগের কথোপকথন:
    {history}
    ---------------------
    প্রাসঙ্গিক তথ্য এবং আগের কথোপকথনের ভিত্তিতে,শুধু বাংলা ভাষায় উsত্তর দিন। 
    প্রশ্ন: {question}
    উত্তর:
    """

    # Get response from Mistral
    answer = run_mistral(prompt)

    # Update conversation history in session
    session['conversation_history'].append(f"User: {question}")
    session['conversation_history'].append(f"Bot: {answer}")

    return jsonify(answer)

if __name__ == "__main__":
    app.run(debug=True)
