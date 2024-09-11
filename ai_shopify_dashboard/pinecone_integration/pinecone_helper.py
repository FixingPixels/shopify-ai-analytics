import pinecone
from ecommerce.shopify_api import fetch_shopify_data
from transformers import pipeline

# Initialize Pinecone and Hugging Face pipeline
pinecone.init(api_key='YOUR_PINECONE_API_KEY', environment='us-west1-gcp')
index = pinecone.Index("ecommerce_data")
qa_pipeline = pipeline('question-answering', model='deepset/roberta-base-squad2')

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
