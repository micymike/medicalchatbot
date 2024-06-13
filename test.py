from flask import Flask, render_template, jsonify, request
from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.llms import CTransformers
from dotenv import load_dotenv
import os
import time
import chromadb
from chromadb.config import Settings

app = Flask(__name__)

# Load environment variables
load_dotenv()

# Initialize ChromaDB client
chroma_client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet"))

# Load data and embeddings
extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

# Setup CTransformers LLM
llm = CTransformers(
    model="llama-2-7b-chat.ggmlv3.q4_0.bin", 
    model_type="llama", 
    config={'max_new_tokens': 256, 'temperature': 0.7}  # Adjusted for performance
)

# List of medical-related keywords for simple filtering
medical_keywords = [
    "disease", "symptom", "treatment", "diagnosis", "medication", "therapy",
    "doctor", "patient", "surgery", "health", "medicine", "medical",
    "infection", "virus", "bacteria", "cancer", "diabetes", "hypertension",
    "cardiology", "neurology", "oncology", "pediatrics", "dermatology"
]

def is_medical_question(question):
    """ Simple keyword-based filter to check if the question is medical. """
    return any(keyword in question.lower() for keyword in medical_keywords)

# Flask routes
@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        msg = request.form["msg"]
        input_text = msg
        print(f"Received message: {input_text}")
        
        # Check if the question is medical
        if not is_medical_question(input_text):
            return "This chatbot only answers medical questions. Please ask a medical-related question."
        
        # Display spinner
        result = {"generated_text": "Thinking..."}
        
        # Simulate processing delay
        time.sleep(1)
        
        # Retrieve response from the model
        result = llm.generate([input_text])
        print(f"LLMResult: {result}")
        
        # Access the generated text from the result object
        if result.generations and result.generations[0]:
            generated_text = result.generations[0][0].text
        else:
            generated_text = "No response generated."
        
        print(f"Response: {generated_text}")
        
        # Here you would add code to interact with ChromaDB if needed
        # For example, to store the conversation or to retrieve similar conversations
        
        return str(generated_text)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)