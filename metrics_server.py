# metrics_server.py - Add this to your trading bot

import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import json
from logger_config import logger
from flask import Flask, jsonify

class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler for metrics endpoint"""
    
    def __init__(self, trading_bot, *args, **kwargs):
        self.trading_bot = trading_bot
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests"""
        path = urlparse(self.path).path
        
        if path == '/metrics':
            self.serve_prometheus_metrics()
        elif path == '/health':
            self.serve_health_check()
        elif path == '/stats':
            self.serve_json_stats()
        else:
            self.send_error(404)
    
    def serve_prometheus_metrics(self):
        """Serve metrics in Prometheus format"""
        try:
            metrics = self.get_trading_metrics()
            
            prometheus_metrics = []
            
            # Trading metrics
            prometheus_metrics.append(f"# HELP trading_bot_balance_btc Current BTC balance")
            prometheus_metrics.append(f"# TYPE trading_bot_balance_btc gauge")
            prometheus_metrics.append(f"trading_bot_balance_btc {metrics.get('btc_balance', 0)}")
            
            prometheus_metrics.append(f"# HELP trading_bot_balance_eur Current EUR balance")
            prometheus_metrics.append(f"# TYPE trading_bot_balance_eur gauge")
            prometheus_metrics.append(f"trading_bot_balance_eur {metrics.get('eur_balance', 0)}")
            
            prometheus_metrics.append(f"# HELP trading_bot_current_price Current BTC price in EUR")
            prometheus_metrics.append(f"# TYPE trading_bot_current_price gauge")
            prometheus_metrics.append(f"trading_bot_current_price {metrics.get('current_price', 0)}")
            
            # Order metrics
            prometheus_metrics.append(f"# HELP trading_bot_pending_orders Number of pending orders")
            prometheus_metrics.append(f"# TYPE trading_bot_pending_orders gauge")
            prometheus_metrics.append(f"trading_bot_pending_orders {metrics.get('pending_orders', 0)}")
            
            prometheus_metrics.append(f"# HELP trading_bot_total_trades_today Total trades executed today")
            prometheus_metrics.append(f"# TYPE trading_bot_total_trades_today counter")
            prometheus_metrics.append(f"trading_bot_total_trades_today {metrics.get('daily_trades', 0)}")
            
            # Performance metrics
            prometheus_metrics.append(f"# HELP trading_bot_profit_margin Current profit margin percentage")
            prometheus_metrics.append(f"# TYPE trading_bot_profit_margin gauge")
            prometheus_metrics.append(f"trading_bot_profit_margin {metrics.get('profit_margin', 0)}")
            
            # Technical indicators
            prometheus_metrics.append(f"# HELP trading_bot_rsi RSI indicator value")
            prometheus_metrics.append(f"# TYPE trading_bot_rsi gauge")
            prometheus_metrics.append(f"trading_bot_rsi {metrics.get('rsi', 0)}")
            
            prometheus_metrics.append(f"# HELP trading_bot_sentiment News sentiment score")
            prometheus_metrics.append(f"# TYPE trading_bot_sentiment gauge")
            prometheus_metrics.append(f"trading_bot_sentiment {metrics.get('sentiment', 0)}")
            
            # System metrics
            prometheus_metrics.append(f"# HELP trading_bot_uptime_seconds Bot uptime in seconds")
            prometheus_metrics.append(f"# TYPE trading_bot_uptime_seconds counter")
            prometheus_metrics.append(f"trading_bot_uptime_seconds {metrics.get('uptime', 0)}")
            
            prometheus_metrics.append(f"# HELP trading_bot_last_update_timestamp Last metrics update timestamp")
            prometheus_metrics.append(f"# TYPE trading_bot_last_update_timestamp gauge")
            prometheus_metrics.append(f"trading_bot_last_update_timestamp {int(time.time())}")
            
            response = '\n'.join(prometheus_metrics) + '\n'
            
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; charset=utf-8')
            self.end_headers()
            self.wfile.write(response.encode('utf-8'))
            
        except Exception as e:
            logger.error(f"Error serving metrics: {e}")
            self.send_error(500)
    
    def serve_health_check(self):
        """Serve health check endpoint"""
        try:
            health_status = {
                "status": "healthy",
                "timestamp": int(time.time()),
                "version": "1.0.0"
            }
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(health_status).encode('utf-8'))
            
        except Exception as e:
            logger.error(f"Error serving health check: {e}")
            self.send_error(500)
    
    def serve_json_stats(self):
        """Serve detailed stats in JSON format"""
        try:
            stats = self.get_trading_metrics()
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(stats, indent=2).encode('utf-8'))
            
        except Exception as e:
            logger.error(f"Error serving stats: {e}")
            self.send_error(500)
    
    def get_trading_metrics(self):
        """Get current trading metrics from the bot"""
        try:
            # Get current balances
            btc_balance = self.trading_bot.trade_executor.get_total_btc_balance() or 0
            eur_balance = self.trading_bot.trade_executor.get_available_balance("EUR") or 0

            # Get current price
            current_price, _ = self.trading_bot.trade_executor.fetch_current_price()
            current_price = current_price or 0
            
            # Get order stats
            pending_orders = len(self.trading_bot.order_manager.get_pending_orders()) if self.trading_bot.order_manager else 0
            
            # Get performance stats
            order_stats = self.trading_bot.order_manager.get_order_statistics() if self.trading_bot.order_manager else {}
            
            # Calculate profit margin
            avg_buy_price = self.trading_bot._estimate_avg_buy_price()
            profit_margin = 0
            if avg_buy_price and current_price:
                profit_margin = ((current_price - avg_buy_price) / avg_buy_price) * 100
            
            # Get uptime
            uptime = time.time() - getattr(self.trading_bot, 'start_time', time.time())
            
            return {
                'btc_balance': btc_balance,
                'eur_balance': eur_balance,
                'current_price': current_price,
                'pending_orders': pending_orders,
                'daily_trades': self.trading_bot.daily_trade_count,
                'profit_margin': profit_margin,
                'rsi': getattr(self.trading_bot, 'last_rsi', 0),
                'sentiment': getattr(self.trading_bot, 'last_sentiment', 0),
                'uptime': uptime,
                'fill_rate': order_stats.get('fill_rate', 0) * 100,
                'total_fees': order_stats.get('total_fees_paid', 0),
                'avg_fill_time': order_stats.get('avg_time_to_fill', 0)
            }
            
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return {}
    
    def log_message(self, format, *args):
        """Suppress HTTP request logging"""
        pass

class MetricsServer:
    """Metrics server for the trading bot"""
    
    def __init__(self, trading_bot, port=8082):
        self.trading_bot = trading_bot
        self.port = port
        self.server = None
        self.server_thread = None
    
    def start(self):
        """Start the metrics server"""
        try:
            # Create handler with trading bot reference
            handler = lambda *args, **kwargs: MetricsHandler(self.trading_bot, *args, **kwargs)
            
            self.server = HTTPServer(('0.0.0.0', self.port), handler)
            self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.server_thread.start()
            
            logger.info(f"Metrics server started on port {self.port}")
            logger.info(f"Metrics available at:")
            logger.info(f"  - http://localhost:{self.port}/metrics (Prometheus)")
            logger.info(f"  - http://localhost:{self.port}/health (Health check)")
            logger.info(f"  - http://localhost:{self.port}/stats (JSON stats)")
            
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
    
    def stop(self):
        """Stop the metrics server"""
        if self.server:
            self.server.shutdown()
            logger.info("Metrics server stopped")

    # HTTP endpoint for risk monitoring (add to metrics_server.py or create new endpoint)
    def add_risk_monitoring_endpoint(self):
        """
        Add this to your metrics server for web-based monitoring.
        """
        f
        
        @self.metrics_server.app.route('/risk')
        def risk_dashboard():
            risk_summary = self.get_risk_summary()
            return jsonify(risk_summary)
        
        @self.metrics_server.app.route('/risk/decisions')
        def recent_decisions():
            import json
            try:
                with open("./risk_decisions.json", 'r') as f:
                    risk_log = json.load(f)
                return jsonify(risk_log[-20:])  # Last 20 decisions
            except Exception as e:
                return jsonify({"error": str(e)})

    # Simple CLI command to check risk status
    def check_risk_status(self):
        """
        Quick CLI command to check current risk status.
        Usage: Add this as a method and call it manually when needed.
        """
        risk_summary = self.get_risk_summary()
        
        print("\n=== RISK STATUS SUMMARY ===")
        print(f"Status: {risk_summary.get('risk_status', 'UNKNOWN')}")
        print(f"24h Decisions: {risk_summary.get('total_decisions_24h', 0)}")
        print(f"Actions: {risk_summary.get('action_breakdown', {})}")
        print(f"Avg Risk Level: {risk_summary.get('avg_risk_off_probability', 0)*100:.1f}%")
        print(f"Max Risk Level: {risk_summary.get('max_risk_off_probability', 0)*100:.1f}%")
        
        latest = risk_summary.get('latest_decision')
        if latest:
            print(f"\nLatest Decision: {latest['action'].upper()} at {latest['timestamp']}")
            print(f"Reasoning: {latest['reasoning']}")
        print("===========================\n")
        