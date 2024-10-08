from django.urls import path
from .views import get_insights
from .views import get_shopify_products, get_shopify_orders,get_shopify_customers

urlpatterns = [
path('get-insights/', get_insights),
path('get_shopify_products/', get_shopify_products,
name='get_shopify_products'),
path('get_shopify_orders/', get_shopify_orders,
name='get_shopify_orders'),
path('get_shopify_customers/', get_shopify_customers,
name='get_shopify_customers'),
]