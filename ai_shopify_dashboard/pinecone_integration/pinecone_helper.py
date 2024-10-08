import os
import pinecone
from ecommerce.shopify_api import fetch_shopify_data
from transformers import pipeline
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


from decouple import config

# Function to load .env variables
def load_env(file_path):
    with open(file_path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue  # Skip comments and empty lines
            key, value = line.strip().split("=", 1)
            os.environ[key] = value.strip('"')

base_dir = os.path.dirname(os.path.abspath(__file__))  # Get the current directory
env_path = os.path.join(base_dir, '.env')
load_env(env_path)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')


# Print the keys to check if they were loaded correctly
print("OPENAI_API_KEY:", OPENAI_API_KEY)
print("PINECONE_API_KEY:", PINECONE_API_KEY)


# Initialize Pinecone using the new method
pc = Pinecone(
    api_key=PINECONE_API_KEY,  # Make sure to set this in your environment
)

# Initialize Pinecone and Hugging Face pipeline
qa_pipeline = pipeline('question-answering', model='deepset/roberta-base-squad2')
# Ensure the index exists, create it if not
if 'ecommerce-data' not in pc.list_indexes().names():
    pc.create_index(
        name='ecommerce-data', 
        dimension=1536,  # Adjust this based on the type of data you're indexing
        metric='euclidean',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
# Access the index
index = pc.Index('ecommerce-data')
# List all indexes to verify
print("pc.list_indexes().names()",pc.list_indexes().names())

# Index Shopify data
def index_shopify_data():
    shopify_data = fetch_shopify_data()
    
    # Prepare vectors for Pinecone indexing
    vectors = []
    for i, product in enumerate(shopify_data):
        vectors.append({
            'id': str(i),
            'values': [product['sales']],  # Index sales data as a simple example
            'metadata': {'text': f"{product['product']} in {product['category']} category with {product['sales']} sales"}
        })
    
    # Upsert (insert/update) data into the Pinecone index
    index.upsert(vectors)

# Query Shopify data using Pinecone and Hugging Face's QA pipeline
def query_with_rag(query):
    # Search the Pinecone index
    pinecone_response = index.query(query, top_k=5, include_metadata=True)
    
    if pinecone_response['matches']:
        context = pinecone_response['matches'][0]['metadata']['text']
    else:
        context = "No relevant data found"
    
    # Generate an AI-powered response using Hugging Face's QA pipeline
    response = qa_pipeline(question=query, context=context)
    return response
