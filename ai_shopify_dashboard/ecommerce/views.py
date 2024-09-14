from rest_framework.decorators import api_view
from rest_framework.response import Response
from pinecone_integration.pinecone_helper import index_shopify_data, query_with_rag

@api_view(['GET'])
def get_insights(request):
    query = request.query_params.get('query', '')

    if query:
        # Index Shopify data if necessary
        index_shopify_data()

        # Get AI-powered insights
        response = query_with_rag(query)
        return Response({"insights": response})
    
    return Response({"error": "No query provided."})

import shopify
from django.shortcuts import render
from django.http import JsonResponse

def connect_shopify():
    shop_url = f"https://{companion-store}.myshopify.com/admin/api/2023-04"
    session = shopify.Session(shop_url, "2023-04", "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")
    shopify.ShopifyResource.activate_session(session)
    return session

