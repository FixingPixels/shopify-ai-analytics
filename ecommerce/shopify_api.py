import shopify

# Shopify API credentials
SHOPIFY_API_KEY = "YOUR_SHOPIFY_API_KEY"
SHOPIFY_PASSWORD = "YOUR_SHOPIFY_PASSWORD"
SHOPIFY_STORE_URL = "https://YOUR_STORE.myshopify.com/admin"

# Initialize Shopify Session
shopify.ShopifyResource.set_site(f"{SHOPIFY_STORE_URL}/api/2023-07")
session = shopify.Session(SHOPIFY_STORE_URL, "2023-07", SHOPIFY_PASSWORD)
shopify.ShopifyResource.activate_session(session)

def fetch_shopify_data():
    products = shopify.Product.find()
    orders = shopify.Order.find()

    # Extract relevant product and order data
    data = []
    for product in products:
        data.append({
            "product": product.title,
            "sales": product.variants[0].inventory_quantity,  # Example sales data
            "category": product.product_type
        })
    return data
