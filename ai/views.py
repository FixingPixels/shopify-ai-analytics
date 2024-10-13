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
            # Prepare a list for exact matches
            exact_matches = []

            # Find all exact matches in product list
            for product in product_list:
                if (product['title'].lower() == query.lower() and
                        product['inventory_quantity'] == 0):  # Change this condition as needed for inventory quantity
                    exact_matches.append(product)

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

            # If there are exact matches, respond immediately
            if exact_matches:
                response_data = []
                for match in exact_matches:
                    availability = 'in stock' if match['inventory_quantity'] > 0 else 'out of stock'
                    response_data.append({
                        "title": match['title'],
                        "availability": availability,
                        "inventory_quantity": match['inventory_quantity'],
                        "price": match['price']
                    })
                return Response({
                    "message": "Exact matches found.",
                    "matches": response_data
                })

            # Search the Pinecone index using the query if no direct match was found
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
                # Check if pipeline response has a valid answer
                if response.get('answer') and response['score'] > 0:
                    # Return the pipeline answer if available and the score is acceptable
                    final_response = {
                        "message": "AI answer with low confidence level",
                        "answer": response['answer']
                    }                
                    # Return the final response
                    fallback_response["related_products"].append(final_response)
                
                # Dynamic handling based on the query context
                if "out of stock" in query.lower() or "currently out of stock" in query.lower():
                    # Only add products that are out of stock
                    for match in pinecone_response['matches']:
                        if match['metadata'].get('inventory_quantity', 0) == 0:
                            fallback_response["related_products"].append(match['metadata']['text'])

                elif "low stock" in query.lower():
                    # Add products that have low stock (for example, below a threshold of 5)
                    low_stock_threshold = 5
                    for match in pinecone_response['matches']:
                        if match['metadata'].get('inventory_quantity', 0) < low_stock_threshold:
                            fallback_response["related_products"].append(match['metadata']['text'])

                elif "in stock" in query.lower():
                    # Only add products that are in stock
                    for match in pinecone_response['matches']:
                        if match['metadata'].get('inventory_quantity', 0) > 0:
                            fallback_response["related_products"].append(match['metadata']['text'])

                elif "available" in query.lower():
                    product_name = query.split("available")[0].strip()  # Extract product name
                    matching_products = [p for p in product_list if product_name.lower() in p['title'].lower()]
                    
                    if matching_products:
                        for product in matching_products:
                            availability = 'in stock' if product['inventory_quantity'] > 0 else 'out of stock'
                            fallback_response["related_products"].append(f"{product['title']} is {availability} with {product['inventory_quantity']} units available.")
                    else:
                        # Extract product names from the query
                        product_names = [word for word in query.split() if word.lower() not in ["available", "is", "the", "in", "stock"]]
                        
                        if product_names:
                            # Look for products that match the names found in the query
                            available_products = []
                            for product in product_list:
                                if any(name.lower() in product['title'].lower() for name in product_names):
                                    # Add the product details if it's available
                                    available_products.append(f"{product['title']} is {'in stock' if product['inventory_quantity'] > 0 else 'out of stock'} with inventory quantity {product['inventory_quantity']}.")

                            # If available products were found, add them to the response
                            if available_products:
                                fallback_response["related_products"].extend(available_products)
                            else:
                                fallback_response["related_products"].append("No products found matching your query.")
                        else:
                            fallback_response["related_products"].append("No specific product mentioned in the query.")
                            
                
                elif "how many" in query.lower() and "available" in query.lower():
                    product_name = query.split("how many")[1].strip()  # Extract product name
                    matching_products = [p for p in product_list if product_name.lower() in p['title'].lower()]
                    
                    if matching_products:
                        for product in matching_products:
                            fallback_response["related_products"].append(f"There are {product['inventory_quantity']} units of {product['title']} available.")
                    else:
                        fallback_response["related_products"].append("No products found matching your query.")



                elif "most expensive" in query.lower():
                    # Identify the most expensive product
                    most_expensive_product = max(product_list, key=lambda x: x['price'], default=None)
                    if most_expensive_product:
                        fallback_response["related_products"].append(f"{most_expensive_product['title']} priced at {most_expensive_product['price']}")

                elif "cheapest" in query.lower():
                    # Identify the cheapest product
                    cheapest_product = min(product_list, key=lambda x: x['price'], default=None)
                    if cheapest_product:
                        fallback_response["related_products"].append(f"{cheapest_product['title']} priced at {cheapest_product['price']}")

                # Handling "best" query
                elif "best" in query.lower():
                    # Logic for determining the best product based on stock, price, or other criteria
                    best_products = []

                    # Option 1: If the best is defined by the highest price (luxury items or premium quality)
                    best_by_price = max(product_list, key=lambda x: x['price'], default=None)
                    if best_by_price:
                        best_products.append(f"{best_by_price['title']} is priced at {best_by_price['price']} and has {best_by_price['inventory_quantity']} units in stock.")

                    # Option 2: If the best is defined by high stock (high demand, well-supplied items)
                    best_by_stock = max(product_list, key=lambda x: x['inventory_quantity'], default=None)
                    if best_by_stock:
                        best_products.append(f"{best_by_stock['title']} has the highest stock of {best_by_stock['inventory_quantity']} units and is priced at {best_by_stock['price']}.")

                    # Option 3: You can also add a condition to combine price and stock if needed
                    if best_by_price and best_by_stock and best_by_price['id'] != best_by_stock['id']:
                        best_products.append(f"For a balance of price and stock: {best_by_stock['title']} is priced at {best_by_stock['price']} and has {best_by_stock['inventory_quantity']} units in stock.")

                    # Option 4: Alternatively, filter for products that have a balance of both criteria.
                    for product in product_list:
                        if product['price'] > 0 and product['inventory_quantity'] > 0:
                            best_products.append(f"{product['title']} is priced at {product['price']} and has {product['inventory_quantity']} units available.")

                    if not best_products:
                        best_products.append("We couldn't find a product that matches the criteria for 'best'.")

                    fallback_response["related_products"].append(best_products)

                
                elif "price" in query.lower():
                    product_name = query.split("price")[1].strip()  # Extract product name
                    matching_products = [p for p in product_list if product_name.lower() in p['title'].lower()]
                    
                    if matching_products:
                        for product in matching_products:
                            fallback_response["related_products"].append(f"The price of {product['title']} is {product['price']}.")
                    else:                       
                        # Extract the product name and find its price
                        for match in pinecone_response['matches']:
                            if match['metadata']['text'].lower().startswith(query.split(" ")[-1].lower()):
                                fallback_response["related_products"].append(f"{match['metadata']['text']} priced at {match['metadata']['price']}")

                elif "compare" in query.lower():
                    # Handle product comparison logic
                    product_names = [word for word in query.split() if word.lower() not in ["compare", "the", "and", "which"]]
                    products_to_compare = [prod for prod in product_list if prod['title'] in product_names]

                    if len(products_to_compare) == 2:
                        comparison_result = f"{products_to_compare[0]['title']} has {products_to_compare[0]['inventory_quantity']} in stock, priced at {products_to_compare[0]['price']}. " \
                                            f"{products_to_compare[1]['title']} has {products_to_compare[1]['inventory_quantity']} in stock, priced at {products_to_compare[1]['price']}."
                        fallback_response["related_products"].append(comparison_result)


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