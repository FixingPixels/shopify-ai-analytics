import pinecone
import openai
from decouple import config
from transformers import pipeline
from rest_framework.decorators import api_view
from rest_framework.response import Response

OPENAI_API_KEY = config('OPENAI_API_KEY')
PINECONE_API_KEY = config('PINECONE_API_KEY')

# Initialize Pinecone and LLM
# pinecone.init(api_key=PINECONE_API_KEY, environment='us-west1-gcp')
pinecone.init(api_key=PINECONE_API_KEY)
index = pinecone.Index("ecommerce-data")
openai.api_key = OPENAI_API_KEY

# Hugging Face LLM (e.g., GPT-3)
qa_pipeline = pipeline('question-answering', model='deepset/roberta-base-squad2')

# Sample data indexed into Pinecone for RAG
# (You would ideally load actual eCommerce data here)
documents = [
    {"product": "Product A", "sales": 150, "category": "Electronics"},
    {"product": "Product B", "sales": 80, "category": "Home Appliances"},
    # Add more products or customer data here
]

# Query the data with RAG
def query_with_rag(query):
    # Search Pinecone index for similar data
    pinecone_response = index.query(query, top_k=5)
    context = pinecone_response['matches'][0]['metadata']['text']

    # Use Hugging Face or OpenAI for context-aware generation
    response = qa_pipeline(question=query, context=context)
    return response

# API to generate insights
@api_view(['GET'])
def get_insights(request):
    query = request.query_params.get('query', '')
    if query:
        response = query_with_rag(query)
        return Response({"insights": response})
    return Response({"error": "No query provided."})
