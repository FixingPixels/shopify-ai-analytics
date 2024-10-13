from rest_framework.decorators import api_view
from rest_framework.response import Response
import shopify
import os
from decouple import config
import requests
from requests.auth import HTTPBasicAuth


# Function to load .env variables
# def load_env(file_path):
#     with open(file_path) as f:
#         for line in f:
#             if line.startswith("#") or not line.strip():
#                 continue  # Skip comments and empty lines
#             key, value = line.strip().split("=", 1)
#             os.environ[key] = value.strip('"')

# base_dir = os.path.dirname(os.path.abspath(__file__))  # Get the current directory
# env_path = os.path.join(base_dir, '.env')
# load_env(env_path)

SHOP_NAME = os.getenv('SHOPIFY_STORE_NAME')
API_KEY = os.getenv('SHOPIFY_API_KEY')
PASSWORD = os.getenv('SHOPIFY_PASSWORD')
SHOP_URL =f"https://{API_KEY}:{PASSWORD}@{SHOP_NAME}/admin/api/2023-04"

url = f'https://{SHOP_NAME}.myshopify.com/admin/api/2023-04/products.json'

response = requests.get(url, auth=HTTPBasicAuth(API_KEY, PASSWORD))

if response.status_code == 200:
    products = response.json()
    print("products",products)
else:
    print(f"Error {response.status_code}: {response.json()}")
    
# Function to set up Shopify session
def shopify_session():
    shop_url =f"https://{API_KEY}:{PASSWORD}@{SHOP_NAME}.myshopify.com/admin"
    print("shop url ",shop_url)
    shopify.ShopifyResource.set_site(shop_url)

    
@api_view(['GET'])
def get_shopify_products(request):
    print(f"Request type - get shopify products: {type(request)}")
    try:
        shopify_session()  # Establish the session with Shopify
        products = shopify.Product.find()  # Fetch products from Shopify
        
        product_list = []
        for product in products:
            product_list.append({
                "id": product.id,
                "title": product.title,
                "inventory_quantity": product.variants[0].inventory_quantity,
                "price": product.variants[0].price
            })
        
        return Response({"products": product_list})
    
    except Exception as e:
        print("Error fetching Shopify products:", str(e))
        return Response({"error": str(e)}, status=500)



# Fetch recent orders from Shopify
@api_view(['GET'])
def get_shopify_orders(request):
    shopify_session()
    orders = shopify.Order.find()
    order_list = []
    for order in orders:
        order_list.append({
            "id": order.id,
            "total_price": order.total_price,
            "customer_email": order.email,
            "line_items": [
                {
                    "product_title": item.title,
                    "quantity": item.quantity,
                    "price": item.price
                }
            for item in order.line_items
            ]
        })
    return Response({"orders": order_list})


@api_view(['GET'])
def get_shopify_customers(request):
    shopify_session()
    customers = shopify.Customer.find()
    customer_list = []
    for customer in customers:
        print(vars(customer))
        customer_list.append({
            "id": customer.id,
            "email": getattr(customer, 'email', 'N/A'),  # Handle missing email attribute
            "first_name": getattr(customer, 'first_name', 'N/A'),
            "last_name": getattr(customer, 'last_name', 'N/A'),
            "orders_count": getattr(customer, 'orders_count', 0)
        })
    return Response({"customers": customer_list})
