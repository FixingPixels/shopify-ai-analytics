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



# Ensure the index exists, create it if not
index_name = 'ecommerce-data-768'
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

@api_view(['GET'])
def get_insights(request):    
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
                    "price": float(product.variants[0].price)
                })
            
            # Check for direct match in product list
            direct_match = next((product for product in product_list if product['title'].lower() in query.lower()), None)

            # Prepare vectors for Pinecone indexing
            vectors = []
            for product in product_list:
                if isinstance(product, dict):
                    vector_value = create_vector_from_product(product)
                    assert len(vector_value) == 768  # Ensure this matches your index
                    vectors.append({
                        'id': str(product['id']),
                        'values': vector_value,
                        'metadata': {
                            'text': f"{product.get('title', 'Unknown product')} with inventory quantity {product.get('inventory_quantity', 0)}",
                            'inventory_quantity': product.get('inventory_quantity', 0),
                            'price': product.get('price', 0)
                        }
                    })

            # Upsert the vectors into Pinecone
            if vectors:
                upsert_response = index.upsert(vectors)

            # If there is a direct match, respond immediately
            if direct_match:
                availability = 'in stock' if direct_match['inventory_quantity'] > 0 else 'out of stock'
                return Response({
                    "message": f"The {direct_match['title']} is {availability} with {direct_match['inventory_quantity']} units available."
                })

            # Search the Pinecone index using the query
            pinecone_response = index.query(vector=create_vector_from_product({'title': query, 'price': '0.00', 'inventory_quantity': 0}), top_k=10, include_metadata=True)
            
            context = " ".join([match['metadata']['text'] for match in pinecone_response['matches']]) if pinecone_response['matches'] else "No relevant data found"
            

            # Generate AI-powered response using Hugging Face's QA pipeline
            response = qa_pipeline(question=query, context=context)
            
            # Determine confidence level
            score = response.get("score", 0)
            confidence = (
                "High Confidence" if score >= 0.8 else
                "Moderate Confidence" if 0.5 <= score < 0.8 else
                "Low Confidence"
            )

            # Fallback handling for low confidence
            if confidence == "Low Confidence":
                fallback_response = {
                    "message": "We couldn't find an exact match, but here are some related products:",
                    "related_products": []
                }

                # Determine related products based on query context
                if "available" in query.lower():
                    product_name = query.split("available")[0].strip()  # Extract product name
                    matching_products = [p for p in product_list if product_name.lower() in p['title'].lower()]
                    
                    for product in matching_products:
                        availability = 'in stock' if product['inventory_quantity'] > 0 else 'out of stock'
                        fallback_response["related_products"].append(f"{product['title']} is {availability} with {product['inventory_quantity']} units available.")

                # Check if related products were found
                if not fallback_response["related_products"]:
                    fallback_response["related_products"].append("No related products found.")

                return Response(fallback_response)

            # If confidence is high or moderate, format the output as usual
            formatted_response = {
                "insights": {
                    "answer": response.get("answer", "I couldn't find an answer."),
                    "score": score,
                    "confidence": confidence
                }
            }
            
            return Response(formatted_response)

        except Exception as general_exception:
            return JsonResponse({'error': str(general_exception)}, status=400)

    return JsonResponse({'success': True})
