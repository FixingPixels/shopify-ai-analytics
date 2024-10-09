import json
import os
import requests
import pinecone
from ecommerce.views import get_shopify_products
from transformers import pipeline, AutoTokenizer, AutoModel
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from rest_framework.decorators import api_view
from django.http import JsonResponse
import torch

from rest_framework.response import Response

# Load environment variables from .env file
load_dotenv()
# Initialize Pinecone and Hugging Face pipeline
qa_pipeline = pipeline('question-answering', model='deepset/roberta-base-squad2')
# Load the Hugging Face model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

def create_vector_from_product(product):
    # Create a string to represent the product
    product_text = f"{product.get('title', 'Unknown product')} priced at {product.get('price', '0.00')} with inventory {product.get('inventory_quantity', 0)}"
    
    # Tokenize and get embeddings
    inputs = tokenizer(product_text, return_tensors='pt')
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1).squeeze().numpy()  # Average token embeddings to get a single vector
    return embeddings

# Initialize Pinecone using the new method
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
pc = Pinecone(api_key=PINECONE_API_KEY)

# Ensure the index exists, create it if not
index_name = 'ecommerce-data'
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name, 
        dimension=768,  # Check the dimensionality of your model output
        metric='euclidean',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

# Access the index
index = pc.Index(index_name)


@api_view(['POST'])
def index_shopify_data(request):
    print(f"Request type: {type(request)}")  # Debugging line
    response = get_shopify_products(request)
    print("response ",response)
    return response # Assuming 'response' is already formatted correctly for DRF

# @api_view(['POST'])
# def index_shopify_data(request):
#     print("Indexing Shopify data...")

#     # Fetch Shopify data
#     response = get_shopify_products(request)  # Make sure this is a Response object
    
#     # Check if the response is valid
#     if response.status_code != 200:
#         return JsonResponse({"error": "Failed to fetch Shopify products"}, status=500)

#     products = response.data.get("products", [])  # Extract products from the Response

#     # Prepare vectors for Pinecone indexing
#     vectors = []
#     for product in products:
#         print(f"Processing product: {product}")  # Debug: print each product
#         if isinstance(product, dict):
#             vector_value = create_vector_from_product(product)  # Create a 768-dimensional vector
#             vectors.append({
#                 'id': str(product['id']),  # Use product ID as the unique ID
#                 'values': vector_value.tolist(),  # Convert numpy array to list
#                 'metadata': {
#                     'text': f"{product.get('title', 'Unknown product')} with inventory quantity {product.get('inventory_quantity', 0)}"
#                 }
#             })
#         else:
#             print(f"Unexpected product format: {product}")  # Catch any unexpected format

#     # Index the vectors into Pinecone
#     if vectors:
#         index.upsert(vectors)  # Upsert vectors into Pinecone
#         print("Vectors indexed successfully.")
    
#     return JsonResponse({"message": "Products indexed successfully."}, status=200)




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

# API to generate insights
@api_view(['GET'])
def get_insights(request):
    print("Received query params:", request.query_params) 
    # query = request.query_params.get('query', '')
    # if query:
    #     # Index Shopify data if necessary
    #     index_shopify_data(request)
    #     response = query_with_rag(query)
    #     return Response({"insights": response})
    # return Response({"error": "No query provided."})
    query_list = request.query_params.getlist('query')  # Use getlist to retrieve all values
    if query_list:
        query = query_list[0]  # Get the first query from the list
        # Index Shopify data if necessary
        index_shopify_data(request)
        response = query_with_rag(query)
        return Response({"insights": response})
    return Response({"error": "No query provided."})

