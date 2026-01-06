from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Input schema for a single customer-level prediction.

    These fields match the customer-level features produced by
    `src.train.build_customer_features` and used by the model.
    """
    CustomerId: str = Field(..., description="Customer identifier (optional for prediction)")
    Transaction_Count: int = Field(..., ge=0, description="Total transaction count for the customer")
    Total_Amount: float = Field(..., description="Sum of transaction amounts for the customer")
    Average_Amount: float = Field(..., description="Average transaction amount for the customer")
    Std_Amount: float = Field(..., ge=0.0, description="Standard deviation of transaction amounts")
    RecencyDays: int = Field(..., ge=0, description="Days since last transaction (higher = less engaged)")


class PredictionResponse(BaseModel):
    """Output schema returned to the user."""
    risk_score: float = Field(..., description="Probability of being high risk (0.0 to 1.0)")
    is_high_risk: bool = Field(..., description="True if risk_score > 0.5")