from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from enum import Enum
import numpy as np
from scipy.optimize import minimize
import logging

app = FastAPI(
    title="Offer Optimizer API",
    description="Optimizes offers based on customer metrics and model probabilities",
    version="1.0.0"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelCategory(str, Enum):
    RETENTION = "retention"
    GROWTH = "growth"
    ACQUISITION = "acquisition"
    OTHER = "other"

class Offer(BaseModel):
    offer_name: str
    price: float = Field(..., gt=0)
    volume: float = Field(..., gt=0)
    conversion_rate: float = Field(..., gt=0, le=1.0)

class ModelInput(BaseModel):
    model_name: str
    model_probability: float = Field(..., ge=0.0, le=1.0)
    model_category: ModelCategory
    available_offers: List[Offer]

class OptimizationRequest(BaseModel):
    customer_id: str
    copcar: float = Field(..., gt=0)
    models: List[ModelInput]

class OptimizationResponse(BaseModel):
    customer_id: str
    copcar: float
    opt_profit: float
    expected_profit: float
    model_name: str
    offer_name: str
    offer_price: float  # Optimized price
    actual_offer_price: float  # Original offer price
    offer_volume: float

class OfferOptimizer:
    def __init__(self):
        self.k = 5.0  # Sigmoid steepness parameter
        self.weights = {
            'profit': 1.0,
            'retention': 1.5,
            'efficiency': 0.3,
            'probability': 0.5
        }

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def calculate_retention_score(self, price: float, copcar: float, churn_prob: float) -> float:
        """Calculate retention score using price-sensitive sigmoid"""
        dilution_rate = 1 - (price/copcar)  # Higher score for more dilution
        return (1 - churn_prob) * self.sigmoid(self.k * dilution_rate)

    def evaluate_offer(self, price: float, offer: Offer, model: ModelInput, copcar: float) -> float:
        """Evaluate a single offer with given price"""
        if model.model_category == ModelCategory.RETENTION:
            # For retention offers, MUST have dilution (price < COPCAR)
            if price >= copcar:
                return float('-inf')
            
            retention_score = self.calculate_retention_score(
                price, copcar, 1 - model.model_probability
            )
            
            # Calculate dilution rate
            dilution_rate = 1 - (price/copcar)
            
            value = (
                self.weights['profit'] * (copcar - price) * offer.conversion_rate +
                self.weights['retention'] * retention_score * copcar +
                self.weights['efficiency'] * (offer.conversion_rate * dilution_rate) +
                self.weights['probability'] * model.model_probability
            )
        else:
            # For growth offers, price must be less than COPCAR
            if price >= copcar:
                return float('-inf')
                
            value = (
                self.weights['profit'] * (copcar - price) * offer.conversion_rate +
                self.weights['efficiency'] * (offer.conversion_rate * copcar/max(price, 0.01)) +
                self.weights['probability'] * model.model_probability
            )
        
        return value

    def optimize_offers(self, request: OptimizationRequest) -> OptimizationResponse:
        """Optimize offer selection and pricing"""
        try:
            best_value = float('-inf')
            best_model = None
            best_offer = None
            best_price = None

            # Evaluate each model and its offers
            for model in request.models:
                for offer in model.available_offers:
                    # For retention offers, ensure original price is less than COPCAR
                    if model.model_category == ModelCategory.RETENTION and offer.price >= request.copcar:
                        continue

                    # Set price bounds based on offer type
                    if model.model_category == ModelCategory.RETENTION:
                        # Retention offers must have dilution
                        price_bounds = (
                            min(request.copcar * 0.3, offer.price * 0.8),  # min price
                            min(request.copcar * 0.9, offer.price)         # max price
                        )
                    else:
                        # Growth offers must be profitable
                        price_bounds = (
                            min(request.copcar * 0.5, offer.price * 0.8),  # min price
                            min(request.copcar * 0.95, offer.price)        # max price
                        )

                    # Initial price guess
                    initial_price = (price_bounds[0] + price_bounds[1]) / 2

                    # Optimize price for this offer
                    result = minimize(
                        lambda p: -self.evaluate_offer(p[0], offer, model, request.copcar),
                        x0=[initial_price],
                        bounds=[price_bounds],
                        method='SLSQP',
                        options={'ftol': 1e-6, 'maxiter': 100}
                    )

                    if result.success:
                        value = -result.fun
                        if value > best_value:
                            best_value = value
                            best_model = model
                            best_offer = offer
                            best_price = result.x[0]

            if not best_offer:
                raise ValueError("No suitable offer found")

            # Calculate expected profit using actual offer price
            expected_profit = request.copcar - best_offer.price

            return OptimizationResponse(
                customer_id=request.customer_id,
                copcar=request.copcar,
                opt_profit=best_value,
                expected_profit=expected_profit,
                model_name=best_model.model_name,
                offer_name=best_offer.offer_name,
                offer_price=best_price,
                actual_offer_price=best_offer.price,
                offer_volume=best_offer.volume
            )

        except Exception as e:
            logger.error(f"Error in optimization for customer {request.customer_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

# Initialize optimizer
optimizer = OfferOptimizer()

@app.post("/optimize", response_model=OptimizationResponse, tags=["Optimization"])
async def optimize_offer(request: OptimizationRequest):
    """
    Optimize offer selection and pricing based on customer metrics and model probabilities.
    
    Parameters:
    - customer_id: Unique identifier for the customer
    - copcar: Customer's COPCAR value
    - models: List of models with their probabilities, categories, and available offers
    
    Returns:
    - Optimized offer details including pricing and expected profits
    """
    return optimizer.optimize_offers(request)

@app.get("/health", tags=["Health"])
async def health_check():
    """Check API health status"""
    return {"status": "healthy"}
