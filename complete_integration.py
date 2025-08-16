#!/usr/bin/env python3
"""
Complete Enhanced Bitcoin Trading Bot Integration
Combines all enhancements: ML learning, peak avoidance, adaptive strategies
"""

import sys
import os
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# Import your existing modules
from trading_bot import TradingBot
from data_manager import DataManager
from trade_executor import TradeExecutor
from onchain_analyzer import OnChainAnalyzer
from order_manager import OrderManager
from config import *
from logger_config import logger

# Import our enhancements
from enhanced_trading_bot import EnhancedTradingBot, enhance_existing_bot
from peak_avoidance_system import PeakAwareTrader
from metrics_server import MetricsServer

class UltimateAdaptiveBot:
    """
    The ultimate adaptive Bitcoin trading bot that:
    1. Learns from every trade using machine learning
    2. Avoids buying at peaks using pattern recognition
    3. Adapts position sizes based on performance and market conditions
    4. Provides detailed analytics and performance tracking
    """
    
    def __init__(self, data_manager, trade_executor, onchain_analyzer, order_manager=None):
        # Initialize the original bot
        self.original_bot = TradingBot(data_manager, trade_executor, onchain_analyzer, order_manager)
        
        # Enhance with ML capabilities
        self.enhanced_bot = enhance_existing_bot(self.original_bot)
        
        # Add peak avoidance system
        self.peak_trader = PeakAwareTrader(lookback_days=90)

        self.metrics_server = MetricsServer(self)
        self.metrics_server.start()
        
        # Performance tracking
        self.session_start_time = datetime.now()
        self.trades_this_session = 0
        self.last_analysis_time = None
        
        # Initialize learning from historical data
        self._initialize_comprehensive_learning()
        
        print("🚀 Ultimate Adaptive Bot initialized!")
        print("📊 Features active: ML Learning, Peak Avoidance, Adaptive Sizing")
    
    def _initialize_comprehensive_learning(self):
        """Initialize all learning systems with historical data"""
        try:
            # Load historical price and indicator data for peak analysis
            print("🧠 Initializing comprehensive learning systems...")
            
            # Get historical OHLC data
            historical_ohlc = self.original_bot.trade_executor.get_ohlc_data(
                pair="BTC/EUR", 
                interval='15m', 
                since=int(time.time() - (90 * 24 * 3600)),  # 90 days
                limit=8640  # 90 days * 96 (15min intervals per day)
            )
            
            if historical_ohlc and len(historical_ohlc) > 100:
                # Prepare data for peak analysis
                historical_data = []
                for candle in historical_ohlc:
                    timestamp, open_p, high, low, close, volume = candle
                    
                    # Calculate basic indicators for each point
                    # (This is simplified - in practice you'd calculate these properly)
                    indicators = {
                        'rsi': 50,  # Would calculate properly from price series
                        'macd': 0,  # Would calculate properly
                        'volume_ratio': 1.0  # Would calculate properly
                    }
                    
                    historical_data.append({
                        'price': close,
                        'volume': volume,
                        'timestamp': datetime.fromtimestamp(timestamp),
                        'indicators': indicators
                    })
                
                # Train peak avoidance system
                self.peak_trader.analyze_historical_data(historical_data)
                print(f"✅ Peak analysis trained on {len(historical_data)} data points")
            else:
                print("⚠️ Limited historical data available for peak analysis")
                
        except Exception as e:
            print(f"⚠️ Error in comprehensive learning initialization: {e}")
    
    def execute_ultimate_strategy(self):
        """
        Execute the ultimate adaptive strategy combining all enhancements
        """
        try:
            start_time = time.time()
            
            # 1. Get market data and indicators
            market_data = self._gather_comprehensive_market_data()
            if not market_data:
                print("❌ Could not gather market data")
                return
            
            # 2. Run peak avoidance analysis
            peak_analysis = self._run_peak_analysis(market_data)
            
            # 3. Generate enhanced ML signal
            enhanced_signal = self.enhanced_bot.generate_enhanced_signal(market_data['indicators'])
            
            # 4. Apply peak avoidance override
            final_decision = self._apply_peak_override(enhanced_signal, peak_analysis, market_data)
            
            # 5. Calculate adaptive position size
            position_info = self._calculate_adaptive_position(final_decision, market_data, peak_analysis)
            
            # 6. Execute trade with all enhancements
            trade_result = self._execute_enhanced_trade(final_decision, position_info, market_data)
            
            # 7. Log comprehensive decision
            self._log_comprehensive_decision(final_decision, peak_analysis, position_info, trade_result, market_data)
            
            # 8. Update learning systems
            self._update_learning_systems(trade_result, market_data)
            
            # 9. Check and manage pending orders
            self.original_bot.check_pending_orders()
            
            execution_time = time.time() - start_time
            print(f"⚡ Strategy execution completed in {execution_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Ultimate strategy execution failed: {e}", exc_info=True)
    
    def _gather_comprehensive_market_data(self) -> Optional[Dict]:
        """Gather all market data needed for decision making"""
        try:
            # Get current price and volume
            current_price, current_volume = self.original_bot.trade_executor.fetch_current_price()
            if not current_price:
                return None
            
            # Get price history
            prices, volumes = self.original_bot.data_manager.load_price_history()
            if len(prices) < 50:
                print("⚠️ Insufficient price history")
                return None
            
            # Calculate technical indicators (reusing original bot logic)
            from indicators import (calculate_rsi, calculate_macd, calculate_bollinger_bands, 
                                  calculate_moving_average, calculate_vwap, fetch_enhanced_news, 
                                  calculate_enhanced_sentiment)
            
            rsi = calculate_rsi(prices) or 50
            macd, signal = calculate_macd(prices) or (0, 0)
            upper_band, ma_short, lower_band = calculate_bollinger_bands(prices) or (current_price, current_price, current_price)
            ma_long = calculate_moving_average(prices, 50) or current_price
            vwap = calculate_vwap(prices, volumes) or current_price
            
            # Get enhanced news analysis
            articles = fetch_enhanced_news(top_n=20)
            news_analysis = calculate_enhanced_sentiment(articles)
            
            # Get on-chain data
            onchain_signals = self.original_bot.onchain_analyzer.get_onchain_signals()
            
            # Get balances
            btc_balance = self.original_bot.trade_executor.get_total_btc_balance() or 0
            eur_balance = self.original_bot.trade_executor.get_available_balance("EUR") or 0
            
            # Get performance metrics
            avg_buy_price = self.original_bot._estimate_avg_buy_price()
            
            return {
                'current_price': current_price,
                'current_volume': current_volume,
                'prices': prices,
                'volumes': volumes,
                'indicators': {
                    'rsi': rsi,
                    'macd': macd,
                    'signal': signal,
                    'ma_short': ma_short,
                    'ma_long': ma_long,
                    'vwap': vwap,
                    'upper_band': upper_band,
                    'lower_band': lower_band,
                    'current_price': current_price,
                    'news_analysis': news_analysis,
                    'sentiment': news_analysis.get('sentiment', 0),
                    'volatility': self.original_bot._calculate_volatility(prices),
                    'avg_buy_price': avg_buy_price,
                    **onchain_signals
                },
                'balances': {
                    'btc': btc_balance,
                    'eur': eur_balance
                },
                'news_analysis': news_analysis,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error gathering market data: {e}")
            return None
    
    def _run_peak_analysis(self, market_data: Dict) -> Dict:
        """Run comprehensive peak avoidance analysis"""
        try:
            current_price = market_data['current_price']
            indicators = market_data['indicators']
            recent_prices = market_data['prices'][-100:]  # Last 100 prices
            recent_volumes = market_data['volumes'][-100:]  # Last 100 volumes
            
            # Check if we should avoid buying due to peak risk
            should_avoid, avoid_reason = self.peak_trader.should_avoid_buying(
                current_price, indicators, recent_prices, recent_volumes
            )
            
            # Get position size adjustment
            base_position = 0.1  # 10% of balance
            adjusted_position, position_reason = self.peak_trader.get_position_adjustment(
                base_position, current_price, indicators, recent_prices, recent_volumes
            )
            
            return {
                'should_avoid_buying': should_avoid,
                'avoid_reason': avoid_reason,
                'position_multiplier': adjusted_position / base_position,
                'position_reason': position_reason,
                'analysis_time': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Peak analysis error: {e}")
            return {
                'should_avoid_buying': False,
                'avoid_reason': 'Peak analysis failed',
                'position_multiplier': 1.0,
                'position_reason': 'Default sizing due to analysis error'
            }
    
    def _apply_peak_override(self, enhanced_signal, peak_analysis: Dict, market_data: Dict) -> Dict:
        """Apply peak avoidance override to enhanced signal"""
        
        original_action = enhanced_signal.action
        final_action = original_action
        override_applied = False
        reasoning = list(enhanced_signal.reasoning)
        
        # Peak avoidance override
        if enhanced_signal.action == 'buy' and peak_analysis['should_avoid_buying']:
            final_action = 'hold'
            override_applied = True
            reasoning.insert(0, f"PEAK OVERRIDE: {peak_analysis['avoid_reason']}")
            
            print(f"🚫 PEAK AVOIDANCE OVERRIDE: Changed BUY to HOLD")
            print(f"   Reason: {peak_analysis['avoid_reason']}")
        
        # Additional safety checks
        current_price = market_data['current_price']
        rsi = market_data['indicators']['rsi']
        risk_off_prob = market_data['news_analysis'].get('risk_off_probability', 0)
        
        # Extreme overbought + high risk-off = never buy
        if (enhanced_signal.action == 'buy' and rsi > 80 and risk_off_prob > 0.7):
            final_action = 'hold'
            override_applied = True
            reasoning.insert(0, "EXTREME RISK: RSI > 80 + High macro risk")
        
        # Adjust confidence if override applied
        final_confidence = enhanced_signal.confidence
        if override_applied:
            final_confidence = max(0.1, final_confidence * 0.5)  # Reduce confidence
        
        return {
            'action': final_action,
            'original_action': original_action,
            'confidence': final_confidence,
            'urgency': enhanced_signal.urgency,
            'risk_level': enhanced_signal.risk_level,
            'reasoning': reasoning,
            'override_applied': override_applied,
            'expected_duration': enhanced_signal.expected_duration
        }
    
    def _calculate_adaptive_position(self, decision: Dict, market_data: Dict, peak_analysis: Dict) -> Dict:
        """Calculate position size with all adaptive factors"""
        
        if decision['action'] not in ['buy', 'sell']:
            return {'position_btc': 0, 'position_eur': 0, 'reasoning': 'No position - holding'}
        
        # Get base position from enhanced bot
        enhanced_position = self.original_bot.decide_amount(
            decision['action'], 
            market_data['indicators'], 
            market_data['balances']['btc'], 
            market_data['balances']['eur']
        )
        
        # Apply peak analysis adjustment
        peak_multiplier = peak_analysis.get('position_multiplier', 1.0)
        
        # Apply confidence adjustment
        confidence_multiplier = 0.5 + (decision['confidence'] * 1.5)  # 0.5x to 2.0x
        
        # Apply recent performance adjustment
        recent_performance = self.enhanced_bot._calculate_recent_performance()
        performance_multiplier = 0.7 + (recent_performance.get('win_rate', 0.5) * 0.6)  # 0.7x to 1.3x
        
        # Calculate final position
        final_multiplier = peak_multiplier * confidence_multiplier * performance_multiplier
        final_position_btc = enhanced_position * final_multiplier

        print(f"🔧 DEBUG: Base position: {enhanced_position:.8f} BTC")
        print(f"🔧 DEBUG: Confidence mult: {confidence_multiplier:.2f}")
        print(f"🔧 DEBUG: Peak mult: {peak_multiplier:.2f}")
        print(f"🔧 DEBUG: Before bounds: {final_position_btc:.8f} BTC")
        print(f"🔧 DEBUG: Performance mult: {performance_multiplier:.2f}")
        print(f"🔧 DEBUG: Final multiplier: {final_multiplier:.2f}")
        print(f"🔧 DEBUG: Final position: {final_position_btc:.8f} BTC")
        print(f"🔧 DEBUG: Recent performance: {recent_performance.get('win_rate', 0.5):.2%}")

        # Apply bounds

        current_price = market_data['current_price']
        available_eur = market_data['balances']['eur']
        available_btc = market_data['balances']['btc']

        print(f"🔧 DEBUG: Current price: €{current_price:.2f}")
        print(f"🔧 DEBUG: Available EUR: €{available_eur:.2f}")
        print(f"🔧 DEBUG: Available BTC: {available_btc:.8f}")


        max_balance = market_data['balances']['eur'] if decision['action'] == 'buy' else market_data['balances']['btc']
        if decision['action'] == 'buy':
            max_spend = available_eur * 0.5  # Use 50% of available EUR
            max_position_btc = (max_balance * 0.25) / market_data['current_price']  # Max 25% of EUR balance
            final_position_btc = max(final_position_btc, 0.0001)
            print(f"🔧 DEBUG: Max spend (50% EUR): €{max_spend:.2f}")
            print(f"🔧 DEBUG: Max position BTC: {max_position_btc:.8f}")
            print(f"🔧 DEBUG: Required EUR for position: €{final_position_btc * current_price:.2f}")

        else:  # sell
            max_position_btc = max_balance * 0.3  # Max 30% of BTC balance
            final_position_btc = min(final_position_btc, max_position_btc)
            print(f"🔧 DEBUG: After max position check: {final_position_btc:.8f} BTC")
        
        # Minimum position check
        min_position_btc = 0.0001  # Minimum trade size
        if final_position_btc < min_position_btc:
            final_position_btc = 0
        
        position_eur = final_position_btc * market_data['current_price']
        
        reasoning = [
            f"Base: {enhanced_position:.6f} BTC",
            f"Peak adj: {peak_multiplier:.2f}x",
            f"Confidence adj: {confidence_multiplier:.2f}x", 
            f"Performance adj: {performance_multiplier:.2f}x",
            f"Final: {final_position_btc:.6f} BTC (€{position_eur:.2f})"
        ]
        
        return {
            'position_btc': final_position_btc,
            'position_eur': position_eur,
            'base_position': enhanced_position,
            'final_multiplier': final_multiplier,
            'reasoning': reasoning
        }
    
    def _execute_enhanced_trade_simple(self, decision: Dict, position_info: Dict, market_data: Dict) -> Dict:
        """Simplified trade execution"""
        
        if decision['action'] not in ['buy', 'sell'] or position_info['position_btc'] <= 0:
            return {'executed': False, 'reason': 'No trade to execute'}
        
        action = decision['action']
        volume = position_info['position_btc']
        
        print(f"🚀 Enhanced {action.upper()}: {volume:.8f} BTC")
        print(f"   Confidence: {decision['confidence']:.1%}")
        
        # Use original bot's trading logic but with our enhanced decisions
        try:
            if self.original_bot.should_wait_for_pending_orders(action):
                return {'executed': False, 'reason': f'Waiting for pending {action} orders'}
            
            order_book = self.original_bot.trade_executor.get_btc_order_book()
            optimal_price = self.original_bot.trade_executor.get_optimal_price(order_book, action)
            
            if optimal_price and self.original_bot.order_manager:
                order_id = self.original_bot.order_manager.place_limit_order_with_timeout(
                    volume=volume,
                    side=action,
                    price=optimal_price,
                    timeout=300
                )
                
                if order_id:
                    self.trades_this_session += 1
                    print(f"✅ Order placed: {order_id}")
                    return {'executed': True, 'action': action, 'volume': volume, 'price': optimal_price}
            
            return {'executed': False, 'reason': 'Could not place order'}
            
        except Exception as e:
            return {'executed': False, 'reason': str(e)}
    
    
    
    def _execute_enhanced_trade(self, decision: Dict, position_info: Dict, market_data: Dict) -> Dict:
        """Execute trade with enhanced order management"""
        
        if decision['action'] not in ['buy', 'sell'] or position_info['position_btc'] <= 0:
            return {'executed': False, 'reason': 'No trade to execute'}
        
        try:
            # Use the enhanced bot's execution logic
            success = self._execute_enhanced_trade_simple(decision, position_info, market_data)
            # success = self.enhanced_bot._execute_enhanced_trade(
            #     enhanced_signal=type('obj', (object,), {
            #         'action': decision['action'],
            #         'confidence': decision['confidence'],
            #         'urgency': decision['urgency'],
            #         'risk_level': decision['risk_level'],
            #         'reasoning': decision['reasoning'],
            #         'expected_duration': decision['expected_duration']
            #     })(),
            #     volume=position_info['position_btc'],
            #     current_price=market_data['current_price']
            # )
            
            if success:
                self.trades_this_session += 1
                return {
                    'executed': True,
                    'action': decision['action'],
                    'volume': position_info['position_btc'],
                    'price': market_data['current_price'],
                    'confidence': decision['confidence'],
                    'timestamp': datetime.now()
                }
            else:
                return {'executed': False, 'reason': 'Trade execution failed'}
                
        except Exception as e:
            logger.error(f"Enhanced trade execution error: {e}")
            return {'executed': False, 'reason': f'Execution error: {str(e)}'}
    
    def _log_comprehensive_decision(self, decision: Dict, peak_analysis: Dict, 
                                  position_info: Dict, trade_result: Dict, market_data: Dict):
        """Log comprehensive decision with all context"""
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'session_trade_number': self.trades_this_session,
            
            # Decision info
            'final_action': decision['action'],
            'original_action': decision.get('original_action'),
            'override_applied': decision.get('override_applied', False),
            'confidence': decision['confidence'],
            'reasoning': decision['reasoning'][:3],  # Top 3 reasons
            
            # Market data
            'price': market_data['current_price'],
            'rsi': market_data['indicators']['rsi'],
            'sentiment': market_data['indicators']['sentiment'],
            'risk_off_prob': market_data['news_analysis'].get('risk_off_probability', 0),
            
            # Peak analysis
            'peak_avoid': peak_analysis['should_avoid_buying'],
            'peak_reason': peak_analysis['avoid_reason'],
            'position_multiplier': peak_analysis['position_multiplier'],
            
            # Position info
            'position_btc': position_info['position_btc'],
            'position_eur': position_info['position_eur'],
            'position_reasoning': position_info['reasoning'][:2],
            
            # Execution result
            'trade_executed': trade_result['executed'],
            'execution_reason': trade_result.get('reason', 'Success')
        }
        
        # Save to ultimate bot log
        log_file = 'ultimate_bot_decisions.json'
        try:
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            logs.append(log_entry)
            logs = logs[-1000:]  # Keep last 1000 entries
            
            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save comprehensive log: {e}")
        
        # Print summary
        print("\n" + "="*60)
        print("🎯 ULTIMATE BOT DECISION SUMMARY")
        print("="*60)
        print(f"💰 Price: €{market_data['current_price']:.2f}")
        print(f"🎬 Action: {decision['action'].upper()}")
        if decision.get('override_applied'):
            print(f"🚫 Override: {decision['original_action'].upper()} → {decision['action'].upper()}")
        print(f"🎯 Confidence: {decision['confidence']:.1%}")
        print(f"📊 Position: {position_info['position_btc']:.6f} BTC (€{position_info['position_eur']:.2f})")
        if peak_analysis['should_avoid_buying']:
            print(f"⚠️ Peak Risk: {peak_analysis['avoid_reason']}")
        print(f"✅ Executed: {trade_result['executed']}")
        print("="*60)
    
    def _update_learning_systems(self, trade_result: Dict, market_data: Dict):
        """Update all learning systems with new data"""
        try:
            # Update peak patterns based on trade outcomes
            # (This would be enhanced with actual trade tracking)
            
            # Update enhanced bot's trade history
            if trade_result['executed']:
                trade_record = {
                    'timestamp': datetime.now(),
                    'action': trade_result['action'],
                    'price': trade_result['price'],
                    'volume': trade_result['volume'],
                    'confidence': trade_result['confidence'],
                    'market_data': market_data['indicators']
                }
                self.enhanced_bot.trade_history.append(trade_record)
            
            # Retrain ML model periodically
            if self.trades_this_session > 0 and self.trades_this_session % 10 == 0:
                print("🧠 Periodic ML model retraining...")
                self.enhanced_bot._retrain_model()
                
        except Exception as e:
            logger.error(f"Error updating learning systems: {e}")
    
    def get_comprehensive_status(self) -> Dict:
        """Get comprehensive status of all bot systems"""
        try:
            # Get basic status
            enhanced_summary = self.enhanced_bot.get_performance_summary()
            
            # Get current market analysis
            market_data = self._gather_comprehensive_market_data()
            if market_data:
                peak_analysis = self._run_peak_analysis(market_data)
            else:
                peak_analysis = {'error': 'Could not gather market data'}
            
            # Session statistics
            session_duration = datetime.now() - self.session_start_time
            
            comprehensive_status = {
                'session_info': {
                    'start_time': self.session_start_time.isoformat(),
                    'duration_hours': session_duration.total_seconds() / 3600,
                    'trades_this_session': self.trades_this_session
                },
                'enhanced_performance': enhanced_summary,
                'current_peak_analysis': peak_analysis,
                'current_market_data': market_data['indicators'] if market_data else None,
                'system_status': {
                    'ml_model_trained': self.enhanced_bot.learning_engine.is_trained,
                    'peak_patterns_count': len(self.peak_trader.peak_system.pattern_database),
                    'trade_history_length': len(self.enhanced_bot.trade_history)
                }
            }
            
            return comprehensive_status
            
        except Exception as e:
            return {'error': f'Status generation failed: {str(e)}'}
    
    def print_comprehensive_status(self):
        """Print detailed status of all systems"""
        print("\n" + "🚀" + "="*58 + "🚀")
        print("🤖 ULTIMATE ADAPTIVE BITCOIN BOT STATUS")
        print("🚀" + "="*58 + "🚀")
        
        try:
            status = self.get_comprehensive_status()
            
            # Session info
            session = status.get('session_info', {})
            print(f"⏰ Session: {session.get('duration_hours', 0):.1f}h, {session.get('trades_this_session', 0)} trades")
            
            # Current market
            if status.get('current_market_data'):
                market = status['current_market_data']
                print(f"💰 BTC: €{market.get('current_price', 0):.2f}")
                print(f"📊 RSI: {market.get('rsi', 50):.1f} | Sentiment: {market.get('sentiment', 0):.3f}")
                
                risk_off = market.get('news_analysis', {}).get('risk_off_probability', 0)
                print(f"⚠️ Macro Risk: {risk_off*100:.0f}%")
            
            # Peak analysis
            if status.get('current_peak_analysis'):
                peak = status['current_peak_analysis']
                if peak.get('should_avoid_buying'):
                    print(f"🚫 PEAK WARNING: {peak.get('avoid_reason', 'High risk detected')}")
                else:
                    print(f"✅ Peak Risk: Low")
                print(f"📏 Position Adj: {peak.get('position_multiplier', 1.0):.2f}x")
            
            # System status
            if status.get('system_status'):
                sys_status = status['system_status']
                print(f"🧠 ML Model: {'Trained' if sys_status.get('ml_model_trained') else 'Learning'}")
                print(f"📚 Peak Patterns: {sys_status.get('peak_patterns_count', 0)}")
                print(f"📈 Trade History: {sys_status.get('trade_history_length', 0)}")
            
            # Performance
            if status.get('enhanced_performance', {}).get('recent_performance'):
                perf = status['enhanced_performance']['recent_performance']
                print(f"🎯 Win Rate: {perf.get('win_rate', 0.5):.1%}")
                print(f"📊 Recent Trades: {perf.get('total_trades', 0)}")
            
            # Current balances
            current_price, _ = self.original_bot.trade_executor.fetch_current_price()
            btc_balance = self.original_bot.trade_executor.get_total_btc_balance() or 0
            eur_balance = self.original_bot.trade_executor.get_available_balance("EUR") or 0
            
            print(f"💎 Holdings: {btc_balance:.6f} BTC | €{eur_balance:.2f} EUR")
            if current_price and btc_balance:
                total_value = eur_balance + (btc_balance * current_price)
                print(f"💰 Total Value: €{total_value:.2f}")
            
        except Exception as e:
            print(f"❌ Error displaying status: {e}")
        
        print("🚀" + "="*58 + "🚀\n")

def main():
    """Main function to run the Ultimate Adaptive Bot"""
    try:
        print("🚀 Initializing Ultimate Adaptive Bitcoin Trading Bot...")
        
        # Initialize your existing components
        # (You'll need to adapt this to your specific setup)
        import ccxt
        from config import BITVAVO_API_KEY, BITVAVO_API_SECRET
        
        # Initialize Bitvavo API
        bitvavo_api = ccxt.bitvavo({
            'apiKey': BITVAVO_API_KEY,
            'secret': BITVAVO_API_SECRET,
            'sandbox': False,  # Set to True for testing
            'enableRateLimit': True,
        })
        
        # Initialize components
        data_manager = DataManager("./price_history.json", "./bot_logs.csv")
        trade_executor = TradeExecutor(bitvavo_api)
        onchain_analyzer = OnChainAnalyzer()
        order_manager = OrderManager(bitvavo_api)
        
        # Create the ultimate bot
        ultimate_bot = UltimateAdaptiveBot(
            data_manager=data_manager,
            trade_executor=trade_executor, 
            onchain_analyzer=onchain_analyzer,
            order_manager=order_manager
        )
        
        print("✅ Ultimate Adaptive Bot ready!")
        ultimate_bot.print_comprehensive_status()
        
        # Main trading loop
        iteration = 0
        while True:
            try:
                iteration += 1
                print(f"\n🔄 Starting iteration {iteration} at {datetime.now().strftime('%H:%M:%S')}")
                
                # Execute the ultimate strategy
                ultimate_bot.execute_ultimate_strategy()
                
                # Print status every 4 iterations (1 hour)
                if iteration % 4 == 0:
                    ultimate_bot.print_comprehensive_status()
                
                # Sleep for 15 minutes
                print("😴 Sleeping for 15 minutes...")
                time.sleep(900)
                
            except KeyboardInterrupt:
                print("\n👋 Shutting down Ultimate Adaptive Bot...")
                break
            except Exception as e:
                logger.error(f"Iteration {iteration} failed: {e}", exc_info=True)
                print(f"❌ Error in iteration {iteration}: {e}")
                print("⏳ Waiting 5 minutes before retry...")
                time.sleep(300)
        
    except Exception as e:
        logger.error(f"Failed to start Ultimate Adaptive Bot: {e}", exc_info=True)
        print(f"💥 Startup failed: {e}")

if __name__ == "__main__":
    main()