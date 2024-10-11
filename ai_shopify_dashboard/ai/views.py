import json
import os
import requests
import pinecone
import shopify
from ecommerce.views import get_shopify_products, shopify_session
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
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModel.from_pretrained('distilbert-base-uncased')

def create_vector_from_product(product):
    # Create a string to represent the product
    product_text = f"{product.get('title', 'Unknown product')} priced at {product.get('price', '0.00')} with inventory {product.get('inventory_quantity', 0)}"
    
    # Tokenize and get embeddings
    inputs = tokenizer(product_text, return_tensors='pt')
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1).squeeze() # Average token embeddings to get a single vector
        
        # Convert embeddings to a Python list directly
        embeddings = embeddings.detach().cpu().numpy()  # Detach tensor and convert to NumPy array        

        # Convert NumPy array to list
        embeddings_list = [float(i) for i in embeddings]  # Convert each element to a float to ensure it's a standard list
        
        return embeddings_list  # Return the Python list d
        

# Initialize Pinecone using the new method
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
pc = Pinecone(api_key=PINECONE_API_KEY)


# List existing indexes and their dimensions
indexes = pc.list_indexes()
for index in indexes:
    print(f"Index: {index.name}, Dimension: {index.dimension}")


# Ensure the index exists, create it if not
index_name = 'ecommerce-data-768'
if index_name not in pc.list_indexes().names():
    print("index name ",index_name)
    pc.create_index(
        name=index_name, 
        dimension=768,  # Check the dimensionality of your model output
        metric='euclidean',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
print ("pc.Index(index_name) ",pc.Index(index_name))
# Access the index
index = pc.Index(index_name)
print("index ",index)

@api_view(['GET'])
def get_insights(request):
    print("Received query params:", request.query_params)
    query = request.query_params.get('query', '')

    if query:
        try:
            # Shopify session and product fetching
            shopify_session()  # Establish the session with Shopify
            products = shopify.Product.find()
            product_list = []

            for product in products:
                product_list.append({
                    "id": product.id,
                    "title": product.title,
                    "inventory_quantity": product.variants[0].inventory_quantity,
                    "price": product.variants[0].price
                })
            print("products ", product_list)

            # Prepare vectors for Pinecone indexing
            vectors = []
            for product in product_list:
                if isinstance(product, dict):
                    vector_value = create_vector_from_product(product)
                    # Check the shape of the vector
                    print(" len(vector_value) ",len(vector_value))
                    assert len(vector_value) == 768  # Ensure this matches your index
                    vectors.append({
                        'id': str(product['id']),
                        'values': vector_value,
                        'metadata': {
                            'text': f"{product.get('title', 'Unknown product')} with inventory quantity {product.get('inventory_quantity', 0)}"
                        }
                    })

            # Upsert the vectors into Pinecone
            if vectors:
                try:
                    upsert_response = index.upsert(vectors)
                    print("Upsert response:", upsert_response)
                except Exception as upsert_exception:
                    print("Exception attributes:", upsert_exception.__dict__)
                    print("Exception type:", type(upsert_exception))

                    
                    print("Error during upsert:", str(upsert_exception))
                    # Check if the exception contains specific error information
                    if hasattr(upsert_exception, 'code'):
                        print("Error code:", upsert_exception.code)
                    if hasattr(upsert_exception, 'message'):
                        print("Error message:", upsert_exception.message)
                    return JsonResponse({'error': str(upsert_exception)}, status=400)
            # Search the Pinecone index using the query
            pinecone_response = index.query(vector=create_vector_from_product({'title': query, 'price': '0.00', 'inventory_quantity': 0}), top_k=5, include_metadata=True)
            print("pinecone_response ",pinecone_response)
            if pinecone_response['matches']:
                context = pinecone_response['matches'][0]['metadata']['text']
            else:
                context = "No relevant data found"
            print("context ",context)
            # Generate AI-powered response using Hugging Face's QA pipeline
            response = qa_pipeline(question=query, context=context)
            print("pipeline response ",response)
            # Format the output
            answer = response.get("answer", "I couldn't find an answer.")
            score = response.get("score", 0)

            formatted_response = {
                "insights": {
                    "answer": answer,
                    "score": score
                }
            }
            print("formatted_response ",formatted_response)
            return Response(formatted_response)


        except Exception as general_exception:
            print("An error occurred:", str(general_exception))
            return JsonResponse({'error': str(general_exception)}, status=400)

    return JsonResponse({'success': True})