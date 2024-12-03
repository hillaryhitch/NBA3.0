import uvicorn
from fastapi.testclient import TestClient
from optimizer_api import app, ModelCategory
import random

client = TestClient(app)

def generate_retention_offers(base_price: float, n_offers: int = 10):
    """Generate realistic retention offers"""
    offers = []
    for i in range(n_offers):
        price_factor = random.uniform(0.8, 1.2)
        volume_factor = random.uniform(0.9, 1.5)
        offers.append({
            "offer_name": f"Retention Offer {i+1}",
            "price": base_price * price_factor,
            "volume": base_price * volume_factor,
            "conversion_rate": random.uniform(0.1, 0.3)
        })
    return offers

def generate_growth_offers(base_price: float, n_offers: int = 10):
    """Generate realistic growth offers"""
    offers = []
    for i in range(n_offers):
        price_factor = random.uniform(0.7, 1.0)  # Growth offers should be below COPCAR
        volume_factor = random.uniform(1.0, 2.0)
        offers.append({
            "offer_name": f"Growth Offer {i+1}",
            "price": base_price * price_factor,
            "volume": base_price * volume_factor,
            "conversion_rate": random.uniform(0.15, 0.35)
        })
    return offers

def test_optimization():
    # Test case: High churn risk customer
    request_data = {
        "customer_id": "CUST001",
        "copcar": 200.0,
        "models": [
            {
                "model_name": "churn_predictor",
                "model_probability": 0.8,
                "model_category": "retention",
                "available_offers": generate_retention_offers(200.0, 12)
            },
            {
                "model_name": "upsell_predictor",
                "model_probability": 0.4,
                "model_category": "growth",
                "available_offers": generate_growth_offers(180.0, 10)
            }
        ]
    }
    
    response = client.post("/optimize", json=request_data)
    print("\nTest Case: High Churn Risk Customer")
    print("Request:")
    print(f"  Customer ID: {request_data['customer_id']}")
    print(f"  COPCAR: ${request_data['copcar']}")
    print("\n  Models and Offers:")
    for model in request_data['models']:
        print(f"\n  {model['model_name']} ({model['model_category']}):")
        print(f"    Probability: {model['model_probability']:.1%}")
        print(f"    Available Offers ({len(model['available_offers'])} offers):")
        for offer in model['available_offers']:
            print(f"      {offer['offer_name']}: ${offer['price']:.2f}, "
                  f"Volume: {offer['volume']:.2f}, Conv: {offer['conversion_rate']:.1%}")
    
    if response.status_code == 200:
        result = response.json()
        print("\nOptimized Result:")
        print(f"  Selected Offer: {result['offer_name']}")
        print(f"  Optimized Price: ${result['offer_price']:.2f}")
        print(f"  Offer Volume: {result['offer_volume']:.2f}")
        print(f"  Expected Profit: ${result['expected_profit']:.2f}")
        print(f"  Optimal Profit: ${result['opt_profit']:.2f}")
        print(f"  Model Used: {result['model_name']}")
    else:
        print("\nError:", response.text)
    print("Status Code:", response.status_code)

    # Test case: Growth opportunity customer
    request_data = {
        "customer_id": "CUST002",
        "copcar": 150.0,
        "models": [
            {
                "model_name": "churn_predictor",
                "model_probability": 0.2,
                "model_category": "retention",
                "available_offers": generate_retention_offers(150.0, 10)
            },
            {
                "model_name": "upsell_predictor",
                "model_probability": 0.7,
                "model_category": "growth",
                "available_offers": generate_growth_offers(140.0, 15)
            }
        ]
    }
    
    response = client.post("/optimize", json=request_data)
    print("\nTest Case: Growth Opportunity Customer")
    print("Request:")
    print(f"  Customer ID: {request_data['customer_id']}")
    print(f"  COPCAR: ${request_data['copcar']}")
    print("\n  Models and Offers:")
    for model in request_data['models']:
        print(f"\n  {model['model_name']} ({model['model_category']}):")
        print(f"    Probability: {model['model_probability']:.1%}")
        print(f"    Available Offers ({len(model['available_offers'])} offers):")
        for offer in model['available_offers']:
            print(f"      {offer['offer_name']}: ${offer['price']:.2f}, "
                  f"Volume: {offer['volume']:.2f}, Conv: {offer['conversion_rate']:.1%}")
    
    if response.status_code == 200:
        result = response.json()
        print("\nOptimized Result:")
        print(f"  Selected Offer: {result['offer_name']}")
        print(f"  Optimized Price: ${result['offer_price']:.2f}")
        print(f"  Offer Volume: {result['offer_volume']:.2f}")
        print(f"  Expected Profit: ${result['expected_profit']:.2f}")
        print(f"  Optimal Profit: ${result['opt_profit']:.2f}")
        print(f"  Model Used: {result['model_name']}")
    else:
        print("\nError:", response.text)
    print("Status Code:", response.status_code)

def test_invalid_requests():
    # Test case: Invalid COPCAR
    request_data = {
        "customer_id": "CUST003",
        "copcar": -100.0,  # Invalid negative COPCAR
        "models": [
            {
                "model_name": "churn_predictor",
                "model_probability": 0.8,
                "model_category": "retention",
                "available_offers": generate_retention_offers(100.0, 10)
            }
        ]
    }
    
    response = client.post("/optimize", json=request_data)
    print("\nTest Case: Invalid COPCAR")
    print("Request:")
    print(f"  Customer ID: {request_data['customer_id']}")
    print(f"  Invalid COPCAR: ${request_data['copcar']}")
    print("\nResponse:", response.text)
    print("Status Code:", response.status_code)

def main():
    print("Testing Offer Optimizer API")
    print("-" * 50)
    
    # Test health endpoint
    response = client.get("/health")
    print("\nHealth Check:")
    print("Response:", response.json())
    print("Status Code:", response.status_code)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Test valid optimization scenarios
    test_optimization()
    
    # Test invalid scenarios
    test_invalid_requests()

if __name__ == "__main__":
    main()
    
    # To run the API server, uncomment:
    # uvicorn.run("optimizer_api:app", host="0.0.0.0", port=8000, reload=True)
