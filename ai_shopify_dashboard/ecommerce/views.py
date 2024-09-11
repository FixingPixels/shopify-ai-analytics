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
