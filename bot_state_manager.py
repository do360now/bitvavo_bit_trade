"""
Bot State Manager - Persist critical data across restarts
Saves: avg_buy_price, total_btc_bought, peak_price, etc.
"""

import json
import os
from datetime import datetime
from logger_config import logger
from typing import Dict, Optional


class BotStateManager:
    """Manages persistent bot state across restarts"""
    
    def __init__(self, state_file: str = "./bot_state.json"):
        self.state_file = state_file
        self.state = self._load_state()
    
    def _load_state(self) -> Dict:
        """Load state from disk"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    logger.info(f"✅ Loaded bot state from {self.state_file}")
                    logger.info(f"   Avg Buy: €{state.get('avg_buy_price', 0):,.2f}")
                    logger.info(f"   Total BTC Bought: {state.get('total_btc_bought', 0):.8f}")
                    return state
            else:
                logger.info(f"No existing state file, starting fresh")
                return self._default_state()
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return self._default_state()
    
    def _default_state(self) -> Dict:
        """Default state for new bot"""
        return {
            'avg_buy_price': 0.0,
            'total_btc_bought': 0.0,
            'peak_price': 0.0,
            'peak_timestamp': 0,
            'last_buy_price': 0.0,
            'last_buy_timestamp': 0,
            'last_sell_price': 0.0,
            'last_sell_timestamp': 0,
            'total_round_trips': 0,
            'total_fees_eur': 0.0,
            'successful_round_trips': 0,
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
        }
    
    def save_state(self) -> bool:
        """Save current state to disk"""
        try:
            self.state['last_updated'] = datetime.now().isoformat()
            
            # Write atomically (write to temp file, then rename)
            temp_file = f"{self.state_file}.tmp"
            with open(temp_file, 'w') as f:
                json.dump(self.state, f, indent=2)
            
            os.replace(temp_file, self.state_file)
            logger.debug(f"State saved to {self.state_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            return False
    
    def update_buy(self, price: float, volume: float, fee_eur: float = 0.0):
        """
        Update state after a buy - NOW WITH FEE TRACKING
        
        Args:
            price: BTC price in EUR
            volume: BTC amount bought
            fee_eur: Trading fee paid in EUR
        """
        # Handle zero volume edge case
        if volume <= 0:
            logger.warning(f"Ignoring buy with zero/negative volume: {volume}")
            return
        
        # Calculate weighted average price INCLUDING fees in cost basis
        total_cost_with_fees = (self.state['avg_buy_price'] * self.state['total_btc_bought']) + \
                               (price * volume) + fee_eur
        self.state['total_btc_bought'] += volume
        self.state['avg_buy_price'] = total_cost_with_fees / self.state['total_btc_bought']
        
        # Track fees
        self.state['total_fees_eur'] += fee_eur
        
        self.state['last_buy_price'] = price
        self.state['last_buy_timestamp'] = datetime.now().timestamp()
        
        logger.info(f"📊 Updated avg buy: €{self.state['avg_buy_price']:,.2f} "
                   f"(total bought: {self.state['total_btc_bought']:.8f} BTC, "
                   f"fees: €{self.state['total_fees_eur']:.2f})")
        
        self.save_state()
    
    def update_sell(self, price: float, volume: float, fee_eur: float = 0.0):
        """
        Update state after a sell - NOW WITH FEE TRACKING
        """
        # Calculate fee if not provided
        if fee_eur == 0.0:
            fee_eur = price * volume * 0.0025  # 0.25% fee

        # Track the sell
        self.state['last_sell_price'] = price
        self.state['last_sell_timestamp'] = datetime.now().timestamp()
        self.state['total_round_trips'] += 1
        self.state['total_fees_eur'] += fee_eur

        # Calculate actual profit (after fees)
        gross_proceeds = price * volume
        net_proceeds = gross_proceeds - fee_eur
        cost_basis = self.state['avg_buy_price'] * volume
        actual_profit = net_proceeds - cost_basis

        # If profitable sell after fees, count as successful
        if actual_profit > 0:
            self.state['successful_round_trips'] += 1

        logger.info(f"📊 Sell recorded at €{price:,.2f}")
        logger.info(f"   Gross: €{gross_proceeds:.2f}, Fee: €{fee_eur:.2f}, Net: €{net_proceeds:.2f}")
        logger.info(f"   Profit: €{actual_profit:.2f} ({actual_profit/cost_basis*100:.2f}%)")
        logger.info(f"   Round trips: {self.state['successful_round_trips']}/{self.state['total_round_trips']} successful")

        self.save_state()

    def check_trailing_stop(self, current_price: float, trail_pct: float = 0.10) -> Optional[float]:
        """
        Check if trailing stop is triggered.

        A trailing stop sells when price drops X% from its all-time peak.

        Args:
            current_price: Current BTC price
            trail_pct: Percentage drop from peak to trigger (default 10%)

        Returns:
            Sell price if trailing stop triggered, None otherwise
        """
        peak = self.state.get('peak_price', 0)

        if peak <= 0 or current_price <= 0:
            return None

        # Update peak if current price is higher
        if current_price > peak:
            self.state['peak_price'] = current_price
            self.state['peak_timestamp'] = datetime.now().timestamp()
            self.save_state()
            return None

        # Calculate drop from peak
        drop_pct = (peak - current_price) / peak

        # Trigger trailing stop if drop exceeds threshold
        if drop_pct >= trail_pct:
            logger.warning(f"🛑 TRAILING STOP TRIGGERED!")
            logger.warning(f"   Peak: €{peak:,.2f} | Current: €{current_price:,.2f}")
            logger.warning(f"   Drop: {drop_pct:.1%} (threshold: {trail_pct:.1%})")
            return current_price

        return None

    def update_peak(self, price: float):
        """Update peak price if current price is higher"""
        if price > self.state.get('peak_price', 0):
            self.state['peak_price'] = price
            self.state['peak_timestamp'] = datetime.now().timestamp()
            self.save_state()

    def get_true_profit_margin(self, current_price: float) -> float:
        """
        Calculate TRUE profit margin including all fees
        
        Returns:
            Profit percentage including fees
        """
        if self.state['avg_buy_price'] <= 0 or self.state['total_btc_bought'] <= 0:
            return 0.0
        
        # avg_buy_price already includes fees in cost basis
        true_cost_per_btc = self.state['avg_buy_price']
        
        # Profit margin on current price vs true cost (including fees)
        profit_margin = ((current_price - true_cost_per_btc) / true_cost_per_btc) * 100
        
        return profit_margin
    
    def update_peak(self, price: float):
        """Update peak price"""
        if price > self.state['peak_price']:
            self.state['peak_price'] = price
            self.state['peak_timestamp'] = datetime.now().timestamp()
            logger.debug(f"New peak: €{price:,.2f}")
            # Save every 10th peak update to avoid too many writes
            if int(price) % 100 == 0:
                self.save_state()
    
    def reset_after_rebuy(self, price: float, volume: float):
        """
        Reset tracking after successful rebuy
        This starts fresh accumulation tracking for next round trip
        """
        # Update with the rebuy
        self.update_buy(price, volume)
        
        # Could optionally reset peak, but probably better to keep historical peak
        # self.state['peak_price'] = 0.0
        
        logger.info(f"🔄 Rebuy complete - new cost basis: €{self.state['avg_buy_price']:,.2f}")
    
    def get_avg_buy_price(self) -> float:
        """Get average buy price"""
        return self.state.get('avg_buy_price', 0.0)
    
    def get_total_btc_bought(self) -> float:
        """Get total BTC bought"""
        return self.state.get('total_btc_bought', 0.0)
    
    def get_peak_price(self) -> float:
        """Get peak price"""
        return self.state.get('peak_price', 0.0)
    
    def get_statistics(self) -> Dict:
        """Get bot statistics - UPDATED with fee info"""
        stats = {
            'avg_buy_price': self.state.get('avg_buy_price', 0.0),
            'total_btc_bought': self.state.get('total_btc_bought', 0.0),
            'total_fees_eur': self.state.get('total_fees_eur', 0.0),  # NEW
            'peak_price': self.state.get('peak_price', 0.0),
            'round_trips': self.state.get('total_round_trips', 0),
            'successful_round_trips': self.state.get('successful_round_trips', 0),
            'win_rate': (
                self.state.get('successful_round_trips', 0) / self.state.get('total_round_trips', 1)
                if self.state.get('total_round_trips', 0) > 0 else 0.0
            ),
            'last_buy_price': self.state.get('last_buy_price', 0.0),
            'last_sell_price': self.state.get('last_sell_price', 0.0),
        }
        return stats
    
    def print_statistics(self):
        """Print current bot statistics"""
        stats = self.get_statistics()
        logger.info("=" * 60)
        logger.info("BOT STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Average Buy Price: €{stats['avg_buy_price']:,.2f}")
        logger.info(f"Total BTC Bought: {stats['total_btc_bought']:.8f}")
        logger.info(f"Peak Price Seen: €{stats['peak_price']:,.2f}")
        logger.info(f"Round Trips: {stats['round_trips']} ({stats['successful_round_trips']} profitable)")
        logger.info(f"Win Rate: {stats['win_rate']:.1%}")
        if stats['last_buy_price'] > 0:
            logger.info(f"Last Buy: €{stats['last_buy_price']:,.2f}")
        if stats['last_sell_price'] > 0:
            logger.info(f"Last Sell: €{stats['last_sell_price']:,.2f}")
        logger.info("=" * 60)
