"""
Models for institutional tracking and whale movement signals
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class InstitutionName(str, Enum):
    """Major institutions we track"""
    BLACKROCK = "BlackRock"
    VANGUARD = "Vanguard" 
    BERKSHIRE_HATHAWAY = "Berkshire Hathaway"
    STATE_STREET = "State Street"
    FIDELITY = "Fidelity"
    JPMORGAN = "JPMorgan"
    BANK_OF_AMERICA = "Bank of America"
    WELLS_FARGO = "Wells Fargo"
    GOLDMAN_SACHS = "Goldman Sachs"
    MORGAN_STANLEY = "Morgan Stanley"


class SignalStrength(str, Enum):
    """Signal strength levels"""
    WEAK = "WEAK"
    MODERATE = "MODERATE" 
    STRONG = "STRONG"
    VERY_STRONG = "VERY_STRONG"


class ActionType(str, Enum):
    """Types of institutional actions"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    INCREASE = "INCREASE"
    DECREASE = "DECREASE"
    NEW_POSITION = "NEW_POSITION"
    LIQUIDATE = "LIQUIDATE"


class InstitutionalHolding(BaseModel):
    """Raw institutional holding data from 13F filings"""
    
    institution: InstitutionName = Field(..., description="Institution name")
    symbol: str = Field(..., description="Stock symbol")
    shares: int = Field(..., description="Number of shares held")
    value: float = Field(..., description="Value of holding in USD")
    filing_date: datetime = Field(..., description="Date of 13F filing")
    quarter: str = Field(..., description="Reporting quarter (e.g., 'Q1 2025')")
    cusip: Optional[str] = Field(None, description="CUSIP identifier")
    percentage_of_portfolio: Optional[float] = Field(None, description="% of institution's total portfolio")


class WhaleMovement(BaseModel):
    """Detected significant movement by a major institution"""
    
    institution: InstitutionName = Field(..., description="Institution making the move")
    symbol: str = Field(..., description="Stock symbol affected")
    action: ActionType = Field(..., description="Type of action taken")
    
    # Change metrics
    shares_change: int = Field(..., description="Change in shares (positive = increase)")
    shares_change_percent: float = Field(..., description="Percentage change in position")
    value_change: float = Field(..., description="Change in USD value")
    
    # Position data
    previous_shares: int = Field(..., description="Previous position size")
    current_shares: int = Field(..., description="Current position size")
    current_value: float = Field(..., description="Current position value in USD")
    
    # Timing and context
    detected_date: datetime = Field(..., description="When this movement was detected")
    filing_date: datetime = Field(..., description="Date of the 13F filing")
    quarter_comparison: str = Field(..., description="Quarters being compared (e.g., 'Q4 2024 vs Q1 2025')")
    
    # Analysis
    signal_strength: SignalStrength = Field(..., description="Strength of the signal")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the signal (0-1)")
    rationale: str = Field(..., description="Reasoning for the signal strength")


class InstitutionalSignal(BaseModel):
    """Final processed signal for AI consumption"""
    
    # Signal identification
    signal_id: str = Field(..., description="Unique identifier for this signal")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When signal was created")
    
    # Primary signal data
    symbol: str = Field(..., description="Stock symbol this signal applies to")
    action_recommendation: ActionType = Field(..., description="Recommended action (BUY/SELL/HOLD)")
    signal_strength: SignalStrength = Field(..., description="Overall signal strength")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in recommendation (0-1)")
    
    # Supporting movements
    whale_movements: List[WhaleMovement] = Field(..., description="List of whale movements supporting this signal")
    
    # Aggregated metrics
    total_institutions_involved: int = Field(..., description="Number of institutions making similar moves")
    net_shares_change: int = Field(..., description="Net change across all institutions")
    net_value_change: float = Field(..., description="Net value change across all institutions")
    dominant_action: ActionType = Field(..., description="Most common action across institutions")
    
    # Context for AI
    ai_summary: str = Field(..., description="Human-readable summary for AI consumption")
    key_institutions: List[InstitutionName] = Field(..., description="Most important institutions in this signal")
    historical_context: Optional[str] = Field(None, description="Historical context if available")
    
    # Risk assessment
    risk_factors: List[str] = Field(default_factory=list, description="Potential risk factors to consider")
    market_cap_impact: Optional[float] = Field(None, description="Estimated market cap impact as percentage")


class InstitutionalAnalysisResult(BaseModel):
    """Result of institutional analysis for multiple symbols"""
    
    analysis_date: datetime = Field(default_factory=datetime.utcnow, description="When analysis was performed")
    symbols_analyzed: List[str] = Field(..., description="Symbols that were analyzed")
    institutions_tracked: List[InstitutionName] = Field(..., description="Institutions included in analysis")
    
    # Generated signals
    signals: List[InstitutionalSignal] = Field(..., description="All institutional signals generated")
    strong_signals: List[InstitutionalSignal] = Field(..., description="Only strong/very strong signals")
    
    # Summary metrics
    total_signals: int = Field(..., description="Total number of signals generated")
    buy_signals: int = Field(..., description="Number of BUY signals")
    sell_signals: int = Field(..., description="Number of SELL signals")
    avg_confidence: float = Field(..., description="Average confidence across all signals")
    
    # For AI consumption
    executive_summary: str = Field(..., description="Executive summary for AI trading decisions")
    top_recommendations: List[str] = Field(..., description="Top 3-5 actionable recommendations")
    
    # Processing metadata
    processing_time_seconds: float = Field(..., description="Time taken to generate analysis")
    data_freshness: str = Field(..., description="How fresh the underlying 13F data is")
    next_update_expected: Optional[datetime] = Field(None, description="When next update is expected")


class InstitutionalConfig(BaseModel):
    """Configuration for institutional tracking"""
    
    # Which institutions to track
    tracked_institutions: List[InstitutionName] = Field(
        default=[InstitutionName.BLACKROCK, InstitutionName.VANGUARD, InstitutionName.BERKSHIRE_HATHAWAY],
        description="Institutions to actively track"
    )
    
    # Thresholds for signal generation
    min_position_change_percent: float = Field(default=5.0, description="Minimum % change to generate signal")
    min_position_value_usd: float = Field(default=100_000_000, description="Minimum position value to track")
    min_confidence_threshold: float = Field(default=0.7, description="Minimum confidence to act on signal")
    
    # Signal strength thresholds
    weak_signal_threshold: float = Field(default=5.0, description="% change for WEAK signal")
    moderate_signal_threshold: float = Field(default=10.0, description="% change for MODERATE signal")
    strong_signal_threshold: float = Field(default=20.0, description="% change for STRONG signal")
    very_strong_signal_threshold: float = Field(default=50.0, description="% change for VERY_STRONG signal")
    
    # Data settings
    lookback_quarters: int = Field(default=2, description="How many quarters back to compare")
    update_frequency_hours: int = Field(default=24, description="How often to check for new data")
    
    # Risk management
    max_signals_per_symbol: int = Field(default=3, description="Max signals to generate per symbol")
    require_multiple_institutions: bool = Field(default=True, description="Require multiple institutions for strong signals")