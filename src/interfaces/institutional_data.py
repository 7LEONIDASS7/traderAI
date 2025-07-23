"""
Interface for fetching institutional data (13F filings, etc.)
Follows the pattern established by other interfaces in the system.
"""

import requests
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, date
import time

from ..utils.exceptions import TradingSystemException
from .. import config


class InstitutionalDataException(TradingSystemException):
    """Exception raised by InstitutionalDataInterface"""
    pass


class InstitutionalDataInterface:
    """
    Interface for fetching institutional holdings data from external APIs.
    Supports 13F filings and institutional tracking.
    """
    
    def __init__(self):
        self.data_source = config.get_string("INSTITUTIONAL_DATA_SOURCE", default="sec_edgar")
        self.base_url = config.get_string("INSTITUTIONAL_DATA_BASE_URL", default="https://data.sec.gov/api/xbrl")
        self.logger = logging.getLogger(__name__)
        
        # Rate limiting for SEC (they require respectful usage)
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests for SEC
        
        self.logger.info("Using FREE SEC EDGAR API for institutional data")
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make HTTP request with rate limiting and error handling
        """
        # SEC EDGAR requires proper headers but no API key
        headers = {
            'User-Agent': 'Trading System Whale Tracker (trading-system@example.com)',
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'data.sec.gov'
        }
        
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        
        try:
            url = f"{self.base_url}/{endpoint}"
            
            self.logger.debug(f"Making SEC EDGAR API request: {url}")
            
            response = requests.get(url, params=params, headers=headers, timeout=30)
            self.last_request_time = time.time()
            
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            error_msg = f"SEC EDGAR API request failed: {str(e)}"
            self.logger.error(error_msg)
            raise InstitutionalDataException(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error in SEC EDGAR request: {str(e)}"
            self.logger.error(error_msg)
            raise InstitutionalDataException(error_msg)
    
    def get_institutional_holders(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Get institutional holders for a specific symbol
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            
        Returns:
            List of institutional holder data
        """
        try:
            endpoint = f"institutional-holder/{symbol}"
            data = self._make_request(endpoint)
            
            if not isinstance(data, list):
                self.logger.warning(f"Unexpected institutional holders data format for {symbol}")
                return []
            
            self.logger.info(f"Retrieved {len(data)} institutional holders for {symbol}")
            return data
            
        except InstitutionalDataException:
            raise
        except Exception as e:
            error_msg = f"Error fetching institutional holders for {symbol}: {str(e)}"
            self.logger.error(error_msg)
            raise InstitutionalDataException(error_msg)
    
    def get_13f_filings(self, cik: str, date: str = None) -> List[Dict[str, Any]]:
        """
        Get 13F filings for a specific institution
        
        Args:
            cik: Central Index Key of the institution
            date: Date in YYYY-MM-DD format (optional, defaults to latest)
            
        Returns:
            List of 13F filing data
        """
        try:
            endpoint = f"form-thirteen/{cik}"
            params = {}
            if date:
                params['date'] = date
            
            data = self._make_request(endpoint, params)
            
            if not isinstance(data, list):
                self.logger.warning(f"Unexpected 13F data format for CIK {cik}")
                return []
            
            self.logger.info(f"Retrieved {len(data)} 13F holdings for CIK {cik}")
            return data
            
        except InstitutionalDataException:
            raise
        except Exception as e:
            error_msg = f"Error fetching 13F filings for CIK {cik}: {str(e)}"
            self.logger.error(error_msg)
            raise InstitutionalDataException(error_msg)
    
    def get_major_institutions(self) -> Dict[str, str]:
        """
        Return mapping of major institution names to their CIK numbers from whale config
        
        Returns:
            Dict mapping institution names to CIK numbers
        """
        import json
        import os
        from .. import config
        
        # Load whale tracking configuration
        whale_config_path = os.path.join(config.PROJECT_ROOT, 'config', 'whale_tracking.json')
        
        institutions = {}
        try:
            if os.path.exists(whale_config_path):
                with open(whale_config_path, 'r') as f:
                    whale_data = json.load(f)
                    
                # Extract CIKs from all tiers
                for tier_name, tier_data in whale_data.get('whales', {}).items():
                    if 'institutions' in tier_data:
                        for inst in tier_data['institutions']:
                            if 'cik' in inst:
                                institutions[inst['name']] = inst['cik']
                                
                self.logger.info(f"Loaded {len(institutions)} whale institutions from config")
            else:
                self.logger.warning("Whale config not found, using default institutions")
        except Exception as e:
            self.logger.error(f"Error loading whale config: {e}")
            
        # Fallback to core institutions if config fails
        if not institutions:
            institutions = {
                "BlackRock": "0001364742",
                "Vanguard": "0000102909", 
                "State Street": "0000093751",
                "Fidelity": "0000315066",
                "Citadel": "0001423053",
                "Renaissance Technologies": "0001037389",
                "Point72": "0001603466",
                "Bridgewater": "0001350694"
            }
            
        return institutions
    
    def get_institution_changes(self, institution_name: str, symbols: List[str], 
                              lookback_quarters: int = 2) -> List[Dict[str, Any]]:
        """
        Get position changes for a major institution across specific symbols
        
        Args:
            institution_name: Name of institution (e.g., 'BlackRock')
            symbols: List of symbols to track
            lookback_quarters: How many quarters back to compare
            
        Returns:
            List of position changes with analysis
        """
        try:
            major_institutions = self.get_major_institutions()
            
            if institution_name not in major_institutions:
                raise InstitutionalDataException(f"Unknown institution: {institution_name}")
            
            cik = major_institutions[institution_name]
            
            # Get latest filings
            filings = self.get_13f_filings(cik)
            
            if not filings:
                self.logger.warning(f"No 13F filings found for {institution_name}")
                return []
            
            # Filter for our target symbols and calculate changes
            changes = []
            for filing in filings[:lookback_quarters]:
                for holding in filing:
                    if holding.get('cusip') and any(symbol.lower() in str(holding.get('nameOfIssuer', '')).lower() 
                                                  for symbol in symbols):
                        changes.append({
                            'institution': institution_name,
                            'symbol': self._extract_symbol(holding),
                            'shares': holding.get('sharesNumber', 0),
                            'value': holding.get('value', 0),
                            'date': filing.get('date'),
                            'quarter': filing.get('period'),
                            'change_type': self._determine_change_type(holding, filings)
                        })
            
            self.logger.info(f"Found {len(changes)} position changes for {institution_name}")
            return changes
            
        except InstitutionalDataException:
            raise
        except Exception as e:
            error_msg = f"Error analyzing institutional changes for {institution_name}: {str(e)}"
            self.logger.error(error_msg)
            raise InstitutionalDataException(error_msg)
    
    def _extract_symbol(self, holding: Dict[str, Any]) -> str:
        """Extract stock symbol from holding data"""
        # This is a simplified extraction - real implementation would need
        # more sophisticated symbol matching
        name = holding.get('nameOfIssuer', '').upper()
        
        # Simple mapping for common companies
        symbol_mapping = {
            'APPLE': 'AAPL',
            'MICROSOFT': 'MSFT', 
            'ALPHABET': 'GOOGL',
            'AMAZON': 'AMZN',
            'TESLA': 'TSLA',
            'META': 'META',
            'SPDR S&P 500': 'SPY',
            'INVESCO QQQ': 'QQQ'
        }
        
        for key, symbol in symbol_mapping.items():
            if key in name:
                return symbol
        
        return holding.get('cusip', 'UNKNOWN')[:4]  # Fallback to partial CUSIP
    
    def _determine_change_type(self, holding: Dict[str, Any], all_filings: List[Dict[str, Any]]) -> str:
        """Determine if this is a buy, sell, or hold"""
        # Simplified logic - real implementation would compare across quarters
        shares = holding.get('sharesNumber', 0)
        
        if shares > 1000000:  # Large position
            return 'BUY'
        elif shares < 100000:  # Small position 
            return 'SELL'
        else:
            return 'HOLD'
    
    def is_available(self) -> bool:
        """Check if institutional data service is available"""
        return True  # SEC EDGAR API is always available for free