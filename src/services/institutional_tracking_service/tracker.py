"""
Institutional Tracking Service - The "Whale Following" Engine

This service:
1. Fetches institutional data via InstitutionalDataInterface  
2. Analyzes whale movements using o3 model
3. Generates actionable signals for o3-pro trading decisions
4. Stores results in MemoryStorage for AI consumption
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import hashlib

from ...interfaces.institutional_data import InstitutionalDataInterface, InstitutionalDataException
from ...interfaces.large_language_model import LLMInterface, LLMException
from ...models.institutional_signal import (
    InstitutionalSignal, WhaleMovement, InstitutionalHolding, 
    InstitutionalAnalysisResult, InstitutionalConfig,
    InstitutionName, ActionType, SignalStrength
)
from ...models.memory_entry import MemoryEntry, MemoryEntryType
from ...services.memory_service.storage import MemoryStorage
from ... import config


class InstitutionalTracker:
    """
    Main service for tracking institutional movements and generating whale following signals
    """
    
    def __init__(self, institutional_data: InstitutionalDataInterface, 
                 llm: LLMInterface, memory_storage: MemoryStorage):
        self.institutional_data = institutional_data
        self.llm = llm
        self.memory_storage = memory_storage
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = self._load_config()
        
        # Model for analysis (use o3 for institutional data processing)
        self.analysis_model = config.get_string("INSTITUTIONAL_TRACKING_LLM_MODEL", default="o3")
        
        self.logger.info("InstitutionalTracker initialized")
    
    def _load_config(self) -> InstitutionalConfig:
        """Load institutional tracking configuration"""
        # Load configuration from environment variables
        return InstitutionalConfig(
            min_position_change_percent=float(config.get_string("INSTITUTIONAL_MIN_CHANGE_PERCENT", default="5.0")),
            min_position_value_usd=float(config.get_string("INSTITUTIONAL_MIN_POSITION_VALUE", default="1000000")),
            min_confidence_threshold=float(config.get_string("INSTITUTIONAL_MIN_CONFIDENCE", default="0.7")),
            lookback_quarters=int(config.get_string("INSTITUTIONAL_LOOKBACK_QUARTERS", default="2"))
        )
    
    def analyze_institutional_movements(self, symbols: List[str]) -> InstitutionalAnalysisResult:
        """
        Main method to analyze institutional movements across given symbols
        
        Args:
            symbols: List of stock symbols to analyze (e.g., ['AAPL', 'MSFT', 'SPY'])
            
        Returns:
            InstitutionalAnalysisResult with all signals and analysis
        """
        start_time = datetime.utcnow()
        self.logger.info(f"Starting institutional analysis for symbols: {symbols}")
        
        try:
            # Step 1: Gather raw institutional data
            raw_data = self._gather_institutional_data(symbols)
            
            # Step 2: Use AI to analyze the data and detect whale movements  
            whale_movements = self._analyze_with_ai(raw_data, symbols)
            
            # Step 3: Generate actionable signals
            signals = self._generate_signals(whale_movements, symbols)
            
            # Step 4: Create analysis result
            analysis_result = self._create_analysis_result(
                symbols, signals, start_time
            )
            
            # Step 5: Store in memory for AI consumption
            self._store_analysis_results(analysis_result)
            
            self.logger.info(f"Completed institutional analysis. Generated {len(signals)} signals.")
            return analysis_result
            
        except Exception as e:
            error_msg = f"Error in institutional analysis: {str(e)}"
            self.logger.error(error_msg)
            
            # Store error in memory
            self._store_error(error_msg, symbols)
            
            # Return empty result rather than crashing
            return InstitutionalAnalysisResult(
                symbols_analyzed=symbols,
                institutions_tracked=list(self.config.tracked_institutions),
                signals=[],
                strong_signals=[],
                total_signals=0,
                buy_signals=0,
                sell_signals=0,
                avg_confidence=0.0,
                executive_summary="Analysis failed due to error",
                top_recommendations=[],
                processing_time_seconds=(datetime.utcnow() - start_time).total_seconds(),
                data_freshness="Error - no data available"
            )
    
    def _gather_institutional_data(self, symbols: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Gather raw institutional data from all tracked institutions"""
        raw_data = {}
        
        for institution in self.config.tracked_institutions:
            try:
                self.logger.debug(f"Fetching data for {institution}")
                
                # Get institution changes for our symbols
                changes = self.institutional_data.get_institution_changes(
                    institution.value, symbols, self.config.lookback_quarters
                )
                
                if changes:
                    raw_data[institution.value] = changes
                    self.logger.info(f"Found {len(changes)} changes for {institution}")
                else:
                    self.logger.warning(f"No data found for {institution}")
                    
            except InstitutionalDataException as e:
                self.logger.warning(f"Failed to fetch data for {institution}: {e}")
                continue
        
        return raw_data
    
    def _analyze_with_ai(self, raw_data: Dict[str, List[Dict[str, Any]]], 
                        symbols: List[str]) -> List[WhaleMovement]:
        """Use AI to analyze raw institutional data and identify whale movements"""
        
        if not raw_data:
            self.logger.warning("No raw institutional data to analyze")
            return []
        
        try:
            # Prepare the prompt for AI analysis
            analysis_prompt = self._create_analysis_prompt(raw_data, symbols)
            
            # Get AI analysis
            self.logger.debug("Sending institutional data to AI for analysis")
            
            response = self.llm.generate_json_response(
                prompt=analysis_prompt,
                model=self.analysis_model,
                max_tokens=2000
            )
            
            # Parse AI response into WhaleMovement objects
            whale_movements = self._parse_ai_response(response)
            
            self.logger.info(f"AI identified {len(whale_movements)} whale movements")
            return whale_movements
            
        except LLMException as e:
            self.logger.error(f"AI analysis failed: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error in AI analysis: {e}")
            return []
    
    def _create_analysis_prompt(self, raw_data: Dict[str, List[Dict[str, Any]]], 
                               symbols: List[str]) -> str:
        """Create prompt for AI to analyze institutional data"""
        
        # Convert raw data to readable format
        data_summary = []
        for institution, changes in raw_data.items():
            for change in changes:
                data_summary.append(f"{institution}: {change}")
        
        prompt = f"""You are an expert institutional analyst. Analyze the following institutional trading data and identify significant "whale movements" that could indicate strong buy/sell signals.

TARGET SYMBOLS: {', '.join(symbols)}

INSTITUTIONAL DATA:
{json.dumps(raw_data, indent=2)}

ANALYSIS REQUIREMENTS:
1. Identify significant position changes (>5% change in holdings)
2. Focus on major institutions: BlackRock, Vanguard, Berkshire Hathaway, etc.
3. Determine if changes indicate BUY, SELL, or HOLD signals
4. Assess signal strength: WEAK, MODERATE, STRONG, VERY_STRONG
5. Provide confidence scores (0.0 to 1.0)

For each significant movement, provide:
- Institution name
- Symbol affected  
- Action type (BUY/SELL/INCREASE/DECREASE/NEW_POSITION/LIQUIDATE)
- Percentage change in position
- Signal strength
- Confidence score
- Rationale for the assessment

Return JSON array of whale movements in this format:
[
  {{
    "institution": "BlackRock",
    "symbol": "AAPL", 
    "action": "INCREASE",
    "shares_change_percent": 15.5,
    "signal_strength": "STRONG",
    "confidence": 0.85,
    "rationale": "BlackRock increased AAPL position by 15.5%, indicating strong bullish sentiment"
  }}
]

Only include movements with confidence >= 0.5 and position changes >= 5%.
"""
        return prompt
    
    def _parse_ai_response(self, response: Dict[str, Any]) -> List[WhaleMovement]:
        """Parse AI response into WhaleMovement objects"""
        whale_movements = []
        
        try:
            # Extract movements from AI response
            if isinstance(response, list):
                movements_data = response
            elif isinstance(response, dict) and 'movements' in response:
                movements_data = response['movements']
            else:
                self.logger.warning(f"Unexpected AI response format: {response}")
                return []
            
            for movement_data in movements_data:
                try:
                    # Create WhaleMovement object
                    movement = WhaleMovement(
                        institution=InstitutionName(movement_data.get('institution')),
                        symbol=movement_data.get('symbol'),
                        action=ActionType(movement_data.get('action')),
                        shares_change=int(movement_data.get('shares_change', 0)),
                        shares_change_percent=float(movement_data.get('shares_change_percent', 0)),
                        value_change=float(movement_data.get('value_change', 0)),
                        previous_shares=int(movement_data.get('previous_shares', 0)),
                        current_shares=int(movement_data.get('current_shares', 0)),
                        current_value=float(movement_data.get('current_value', 0)),
                        detected_date=datetime.utcnow(),
                        filing_date=datetime.utcnow() - timedelta(days=45),  # 13F filings are ~45 days old
                        quarter_comparison=movement_data.get('quarter_comparison', 'Q4 2024 vs Q1 2025'),
                        signal_strength=SignalStrength(movement_data.get('signal_strength', 'WEAK')),
                        confidence=float(movement_data.get('confidence', 0.5)),
                        rationale=movement_data.get('rationale', 'AI detected institutional movement')
                    )
                    
                    whale_movements.append(movement)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to parse movement data {movement_data}: {e}")
                    continue
            
            return whale_movements
            
        except Exception as e:
            self.logger.error(f"Failed to parse AI response: {e}")
            return []
    
    def _generate_signals(self, whale_movements: List[WhaleMovement], 
                         symbols: List[str]) -> List[InstitutionalSignal]:
        """Generate actionable trading signals from whale movements"""
        
        signals = []
        
        # Group movements by symbol
        movements_by_symbol = {}
        for movement in whale_movements:
            if movement.symbol not in movements_by_symbol:
                movements_by_symbol[movement.symbol] = []
            movements_by_symbol[movement.symbol].append(movement)
        
        # Generate signal for each symbol with movements
        for symbol, movements in movements_by_symbol.items():
            try:
                signal = self._create_signal_for_symbol(symbol, movements)
                if signal and signal.confidence >= self.config.min_confidence_threshold:
                    signals.append(signal)
                    self.logger.info(f"Generated {signal.signal_strength} signal for {symbol}")
                    
            except Exception as e:
                self.logger.warning(f"Failed to generate signal for {symbol}: {e}")
                continue
        
        return signals
    
    def _create_signal_for_symbol(self, symbol: str, movements: List[WhaleMovement]) -> Optional[InstitutionalSignal]:
        """Create aggregated signal for a specific symbol"""
        
        if not movements:
            return None
        
        # Filter movements by position value threshold
        significant_movements = []
        for movement in movements:
            if movement.current_value >= self.config.min_position_value_usd:
                significant_movements.append(movement)
            else:
                self.logger.debug(f"Filtered out {movement.institution.value} position in {symbol}: "
                                f"${movement.current_value:,.0f} < ${self.config.min_position_value_usd:,.0f}")
        
        if not significant_movements:
            self.logger.debug(f"No significant movements for {symbol} above ${self.config.min_position_value_usd:,.0f} threshold")
            return None
        
        # Calculate aggregated metrics from significant movements only
        total_institutions = len(set(m.institution for m in significant_movements))
        net_shares_change = sum(m.shares_change for m in significant_movements)
        net_value_change = sum(m.value_change for m in significant_movements)
        avg_confidence = sum(m.confidence for m in significant_movements) / len(significant_movements)
        
        # Determine dominant action
        buy_actions = sum(1 for m in significant_movements if m.action in [ActionType.BUY, ActionType.INCREASE, ActionType.NEW_POSITION])
        sell_actions = sum(1 for m in significant_movements if m.action in [ActionType.SELL, ActionType.DECREASE, ActionType.LIQUIDATE])
        
        if buy_actions > sell_actions:
            dominant_action = ActionType.BUY
        elif sell_actions > buy_actions:
            dominant_action = ActionType.SELL  
        else:
            dominant_action = ActionType.HOLD
        
        # Determine overall signal strength
        max_strength = max(m.signal_strength for m in movements)
        if len(movements) > 2:  # Multiple institutions agree
            overall_strength = max_strength
        else:
            # Downgrade if only one institution
            strength_map = {
                SignalStrength.VERY_STRONG: SignalStrength.STRONG,
                SignalStrength.STRONG: SignalStrength.MODERATE,
                SignalStrength.MODERATE: SignalStrength.WEAK,
                SignalStrength.WEAK: SignalStrength.WEAK
            }
            overall_strength = strength_map.get(max_strength, SignalStrength.WEAK)
        
        # Create AI summary
        key_institutions = list(set(m.institution for m in movements))
        ai_summary = self._create_ai_summary(symbol, movements, dominant_action, key_institutions)
        
        # Create signal
        signal = InstitutionalSignal(
            signal_id=self._generate_signal_id(symbol, movements),
            symbol=symbol,
            action_recommendation=dominant_action,
            signal_strength=overall_strength,
            confidence=avg_confidence,
            whale_movements=movements,
            total_institutions_involved=total_institutions,
            net_shares_change=net_shares_change,
            net_value_change=net_value_change,
            dominant_action=dominant_action,
            ai_summary=ai_summary,
            key_institutions=key_institutions,
            risk_factors=self._assess_risk_factors(movements)
        )
        
        return signal
    
    def _create_ai_summary(self, symbol: str, movements: List[WhaleMovement], 
                          dominant_action: ActionType, key_institutions: List[InstitutionName]) -> str:
        """Create human-readable summary for AI consumption"""
        
        institution_names = [inst.value for inst in key_institutions]
        
        if dominant_action == ActionType.BUY:
            return f"BULLISH: {', '.join(institution_names)} increased positions in {symbol}. {len(movements)} institutional buy signals detected."
        elif dominant_action == ActionType.SELL:
            return f"BEARISH: {', '.join(institution_names)} decreased positions in {symbol}. {len(movements)} institutional sell signals detected."
        else:
            return f"NEUTRAL: Mixed institutional activity in {symbol}. Monitor for clearer signals."
    
    def _assess_risk_factors(self, movements: List[WhaleMovement]) -> List[str]:
        """Assess potential risk factors"""
        risk_factors = []
        
        # Check for conflicting signals
        actions = set(m.action for m in movements)
        if ActionType.BUY in actions and ActionType.SELL in actions:
            risk_factors.append("Conflicting institutional signals - some buying, some selling")
        
        # Check confidence spread
        confidences = [m.confidence for m in movements]
        if max(confidences) - min(confidences) > 0.3:
            risk_factors.append("Wide confidence spread across institutional movements")
        
        # Check if only one institution
        if len(set(m.institution for m in movements)) == 1:
            risk_factors.append("Signal based on single institution - consider waiting for confirmation")
        
        return risk_factors
    
    def _generate_signal_id(self, symbol: str, movements: List[WhaleMovement]) -> str:
        """Generate unique ID for signal"""
        movement_data = f"{symbol}_{len(movements)}_{datetime.utcnow().isoformat()}"
        return hashlib.md5(movement_data.encode()).hexdigest()[:12]
    
    def _create_analysis_result(self, symbols: List[str], signals: List[InstitutionalSignal],
                               start_time: datetime) -> InstitutionalAnalysisResult:
        """Create final analysis result"""
        
        strong_signals = [s for s in signals if s.signal_strength in [SignalStrength.STRONG, SignalStrength.VERY_STRONG]]
        buy_signals = len([s for s in signals if s.action_recommendation == ActionType.BUY])
        sell_signals = len([s for s in signals if s.action_recommendation == ActionType.SELL])
        avg_confidence = sum(s.confidence for s in signals) / len(signals) if signals else 0.0
        
        # Create executive summary
        executive_summary = self._create_executive_summary(signals, strong_signals)
        
        # Top recommendations
        top_recommendations = self._create_top_recommendations(strong_signals)
        
        return InstitutionalAnalysisResult(
            symbols_analyzed=symbols,
            institutions_tracked=list(self.config.tracked_institutions),
            signals=signals,
            strong_signals=strong_signals,
            total_signals=len(signals),
            buy_signals=buy_signals,
            sell_signals=sell_signals,
            avg_confidence=avg_confidence,
            executive_summary=executive_summary,
            top_recommendations=top_recommendations,
            processing_time_seconds=(datetime.utcnow() - start_time).total_seconds(),
            data_freshness="Latest 13F filings (up to 45 days old)"
        )
    
    def _create_executive_summary(self, signals: List[InstitutionalSignal], 
                                 strong_signals: List[InstitutionalSignal]) -> str:
        """Create executive summary for AI"""
        
        if not signals:
            return "No significant institutional movements detected."
        
        buy_count = len([s for s in signals if s.action_recommendation == ActionType.BUY])
        sell_count = len([s for s in signals if s.action_recommendation == ActionType.SELL])
        
        summary = f"Institutional Analysis: {len(signals)} signals detected ({len(strong_signals)} strong). "
        summary += f"{buy_count} BUY signals, {sell_count} SELL signals. "
        
        if strong_signals:
            strong_symbols = [s.symbol for s in strong_signals]
            summary += f"Strong institutional activity in: {', '.join(strong_symbols)}."
        
        return summary
    
    def _create_top_recommendations(self, strong_signals: List[InstitutionalSignal]) -> List[str]:
        """Create top actionable recommendations"""
        recommendations = []
        
        for signal in strong_signals[:5]:  # Top 5 recommendations
            action = signal.action_recommendation.value
            recommendations.append(f"{action} {signal.symbol} - {signal.ai_summary}")
        
        if not recommendations:
            recommendations.append("No strong institutional signals at this time - monitor for updates")
        
        return recommendations
    
    def _store_analysis_results(self, analysis_result: InstitutionalAnalysisResult):
        """Store analysis results in memory for AI consumption"""
        try:
            # Store overall analysis
            analysis_entry = MemoryEntry(
                entry_type=MemoryEntryType.INSTITUTIONAL_DATA,
                source_service="InstitutionalTracker", 
                payload={
                    "analysis_type": "institutional_analysis",
                    "symbols_analyzed": analysis_result.symbols_analyzed,
                    "total_signals": analysis_result.total_signals,
                    "executive_summary": analysis_result.executive_summary,
                    "top_recommendations": analysis_result.top_recommendations,
                    "analysis_timestamp": analysis_result.analysis_date.isoformat()
                }
            )
            self.memory_storage.save_memory(analysis_entry)
            
            # Store each strong signal individually for easy retrieval
            for signal in analysis_result.strong_signals:
                signal_entry = MemoryEntry(
                    entry_type=MemoryEntryType.INSTITUTIONAL_SIGNAL,
                    source_service="InstitutionalTracker",
                    payload={
                        "signal_id": signal.signal_id,
                        "symbol": signal.symbol,
                        "action": signal.action_recommendation.value,
                        "strength": signal.signal_strength.value,
                        "confidence": signal.confidence,
                        "ai_summary": signal.ai_summary,
                        "key_institutions": [inst.value for inst in signal.key_institutions],
                        "risk_factors": signal.risk_factors
                    }
                )
                self.memory_storage.save_memory(signal_entry)
            
            self.logger.info(f"Stored {len(analysis_result.strong_signals)} institutional signals in memory")
            
        except Exception as e:
            self.logger.error(f"Failed to store analysis results: {e}")
    
    def _store_error(self, error_msg: str, symbols: List[str]):
        """Store error in memory"""
        try:
            error_entry = MemoryEntry(
                entry_type=MemoryEntryType.ERROR,
                source_service="InstitutionalTracker",
                payload={
                    "error_type": "institutional_analysis_error",
                    "error_message": error_msg,
                    "symbols": symbols,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            self.memory_storage.save_memory(error_entry)
        except Exception as e:
            self.logger.error(f"Failed to store error in memory: {e}")
    
    def get_recent_institutional_signals(self, symbols: Optional[List[str]] = None,
                                       hours_back: int = 24) -> List[Dict[str, Any]]:
        """
        Retrieve recent institutional signals from memory
        
        Args:
            symbols: Optional list of symbols to filter by
            hours_back: How many hours back to look
            
        Returns:
            List of institutional signal data for AI consumption
        """
        try:
            # Query memory for recent institutional signals
            recent_memories = self.memory_storage.query_memories(
                entry_types=[MemoryEntryType.INSTITUTIONAL_SIGNAL],
                hours_back=hours_back,
                limit=50
            )
            
            signals = []
            for memory in recent_memories:
                payload = memory.payload
                
                # Filter by symbols if provided
                if symbols and payload.get('symbol') not in symbols:
                    continue
                
                signals.append(payload)
            
            self.logger.info(f"Retrieved {len(signals)} recent institutional signals")
            return signals
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve recent institutional signals: {e}")
            return []
    
    def is_available(self) -> bool:
        """Check if institutional tracking is available"""
        return self.institutional_data.is_available()