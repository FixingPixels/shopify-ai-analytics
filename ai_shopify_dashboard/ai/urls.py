from django.urls import path
from .views import get_insights, index_shopify_data


urlpatterns = [    
    path('get_insights/', get_insights,name='get_insights'),
    path('index_shopify_data/', index_shopify_data,name='index_shopify_data')
]