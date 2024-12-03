# Offer Optimization API

A FastAPI-based service that optimizes offer selection and pricing based on customer metrics, model probabilities, and business constraints.

## Mathematical Formulation

### Objective Function

For each offer, we calculate a score that combines multiple factors:

```python
score = w1 * profit_term + w2 * retention_term + w3 * efficiency_term + w4 * probability_term
```

Where:
- `w1, w2, w3, w4` are configurable weights
- Terms vary based on offer category (retention vs growth)

#### For Retention Offers:
```python
profit_term = (COPCAR - price) * conversion_rate
retention_term = retention_score * COPCAR
efficiency_term = conversion_rate * dilution_rate
probability_term = model_probability
```

Where:
- `retention_score = (1 - churn_probability) * sigmoid(k * dilution_rate)` #lower offer prices means higher retention
- `dilution_rate = 1 - (price/COPCAR)`
- Must have `price < COPCAR` (mandatory dilution)

#### For Growth Offers:
```python
profit_term = (COPCAR - price) * conversion_rate
efficiency_term = conversion_rate * COPCAR/price
probability_term = model_probability
```

### Constraints

1. Retention Offers:
   - Price must be below COPCAR
   - Price bounds: `[0.3 * COPCAR, 0.9 * COPCAR]`
   - Dilution rate: 10-70%

2. Growth Offers:
   - Price must be below COPCAR
   - Price bounds: `[0.5 * COPCAR, 0.95 * COPCAR]`

3. All Offers:
   - Conversion rates: `0 < rate ≤ 1`
   - Prices must be positive

## API Usage

### Endpoint

```
POST /optimize
```

### Request Format

```json
{
  "customer_id": "string",
  "copcar": float,
  "models": [
    {
      "model_name": "string",
      "model_probability": float,
      "model_category": "retention" | "growth",
      "available_offers": [
        {
          "offer_name": "string",
          "price": float,
          "volume": float,
          "conversion_rate": float
        }
      ]
    }
  ]
}
```

### Response Format

```json
{
  "customer_id": "string",
  "copcar": float,
  "opt_profit": float,
  "expected_profit": float,
  "model_name": "string",
  "offer_name": "string",
  "offer_price": float,
  "actual_offer_price": float,
  "offer_volume": float
}
```

Where:
- `offer_price`: Optimized price
- `actual_offer_price`: Original offer price
- `expected_profit`: COPCAR - actual_offer_price
- `opt_profit`: Optimization score

## Example Usage

```python
import requests

request_data = {
    "customer_id": "CUST001",
    "copcar": 200.0,
    "models": [
        {
            "model_name": "churn_predictor",
            "model_probability": 0.8,
            "model_category": "retention",
            "available_offers": [
                {
                    "offer_name": "Retention Offer 1",
                    "price": 150.0,
                    "volume": 200.0,
                    "conversion_rate": 0.15
                }
            ]
        }
    ]
}

response = requests.post(
    "http://localhost:8000/optimize",
    json=request_data
)

if response.status_code == 200:
    result = response.json()
    print(f"Selected Offer: {result['offer_name']}")
    print(f"Original Price: ${result['actual_offer_price']}")
    print(f"Optimized Price: ${result['offer_price']}")
    print(f"Expected Profit: ${result['expected_profit']}")
```

## Docker Deployment

1. Build the image:
```bash
docker build -t offer-optimizer .
```

2. Run the container:
```bash
docker run -p 8000:8000 offer-optimizer
```

3. Access the API at `http://localhost:8000`

## Development Setup

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

2. Install dependencies:
```bash
pip install fastapi uvicorn scipy numpy
```

3. Run the server:
```bash
uvicorn optimizer_api:app --reload
```

## Testing

Run the test script:
```bash
python test_api_live.py
```

This will run test scenarios including:
- High churn risk customer
- Growth opportunity customer
- Invalid request handling
