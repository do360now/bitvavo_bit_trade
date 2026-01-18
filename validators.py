"""
Input Validation Utilities
"""

from typing import Union
from logger_config import logger


class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass


class TradingValidator:
    """Validates trading inputs to prevent errors"""
    
    # Bitvavo limits (adjust for your exchange)
    MIN_BTC_ORDER = 0.0001  # 0.0001 BTC minimum
    MIN_EUR_ORDER = 5.0     # €5 minimum order value
    MAX_BTC_ORDER = 100.0   # Sanity check - adjust as needed
    
    @staticmethod
    def validate_price(price: Union[int, float], asset: str = "BTC") -> float:
        """
        Validate price is positive and reasonable
        
        Args:
            price: Price to validate
            asset: Asset name for error messages
        
        Returns:
            Validated price as float
        
        Raises:
            ValidationError: If price is invalid
        """
        if not isinstance(price, (int, float)):
            raise ValidationError(f"Price must be numeric, got {type(price)}")
        
        if price <= 0:
            raise ValidationError(f"Price must be positive, got {price}")
        
        if price != price:  # Check for NaN
            raise ValidationError(f"Price is NaN")
        
        # Sanity checks
        if asset == "BTC":
            if price < 1000:
                raise ValidationError(f"BTC price {price} seems unrealistically low")
            if price > 1000000:
                raise ValidationError(f"BTC price {price} seems unrealistically high")
        
        return float(price)
    
    @staticmethod
    def validate_volume(volume: Union[int, float], asset: str = "BTC") -> float:
        """
        Validate volume is positive and within limits
        
        Args:
            volume: Volume to validate
            asset: Asset name for limits
        
        Returns:
            Validated volume as float
        
        Raises:
            ValidationError: If volume is invalid
        """
        if not isinstance(volume, (int, float)):
            raise ValidationError(f"Volume must be numeric, got {type(volume)}")
        
        if volume <= 0:
            raise ValidationError(f"Volume must be positive, got {volume}")
        
        if volume != volume:  # Check for NaN
            raise ValidationError(f"Volume is NaN")
        
        if asset == "BTC":
            if volume < TradingValidator.MIN_BTC_ORDER:
                raise ValidationError(
                    f"BTC volume {volume:.8f} below minimum {TradingValidator.MIN_BTC_ORDER}"
                )
            
            if volume > TradingValidator.MAX_BTC_ORDER:
                raise ValidationError(
                    f"BTC volume {volume:.8f} exceeds maximum {TradingValidator.MAX_BTC_ORDER}"
                )
        
        return float(volume)
    
    @staticmethod
    def validate_order(volume: float, price: float, side: str, asset: str = "BTC") -> tuple:
        """
        Validate complete order parameters
        
        Args:
            volume: Order volume
            price: Order price
            side: 'buy' or 'sell'
            asset: Asset being traded
        
        Returns:
            (validated_volume, validated_price, validated_side)
        
        Raises:
            ValidationError: If any parameter is invalid
        """
        # Validate side
        if side not in ['buy', 'sell']:
            raise ValidationError(f"Invalid side: {side}, must be 'buy' or 'sell'")
        
        # Validate price and volume
        validated_price = TradingValidator.validate_price(price, asset)
        validated_volume = TradingValidator.validate_volume(volume, asset)
        
        # Validate order value
        order_value = validated_volume * validated_price
        if order_value < TradingValidator.MIN_EUR_ORDER:
            raise ValidationError(
                f"Order value €{order_value:.2f} below minimum €{TradingValidator.MIN_EUR_ORDER}"
            )
        
        return validated_volume, validated_price, side
    
    @staticmethod
    def validate_percentage(value: float, min_pct: float = 0.0, max_pct: float = 100.0) -> float:
        """
        Validate percentage value
        
        Args:
            value: Percentage value to validate
            min_pct: Minimum allowed percentage
            max_pct: Maximum allowed percentage
        
        Returns:
            Validated percentage
        
        Raises:
            ValidationError: If percentage is invalid
        """
        if not isinstance(value, (int, float)):
            raise ValidationError(f"Percentage must be numeric, got {type(value)}")
        
        if value != value:  # NaN check
            raise ValidationError(f"Percentage is NaN")
        
        if value < min_pct or value > max_pct:
            raise ValidationError(
                f"Percentage {value} outside valid range [{min_pct}, {max_pct}]"
            )
        
        return float(value)