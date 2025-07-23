import time
import schedule # Using schedule library for easier task scheduling
import signal
import sys
from datetime import datetime, timezone
from typing import Optional

from ... import config
from ...utils.logger import log
from ...utils.exceptions import (
    TradingSystemError, ConfigError, BrokerageError, LLMError,
    MemoryServiceError, AIServiceError, ExecutionServiceError,
    OptimizationServiceError, NotificationError
)
from ...interfaces.brokerage import BrokerageInterface
from ...interfaces.large_language_model import LLMInterface
from ...interfaces.notification import NotificationInterface
from ...interfaces.institutional_data import InstitutionalDataInterface
from ..memory_service.storage import MemoryStorage
from ..memory_service.organizer import MemoryOrganizer
from ..ai_service.processor import AIServiceProcessor
from ..execution_service.manager import ExecutionServiceManager
from ..optimization_service.engine import OptimizationEngine
from ..optimization_service.frequency_analyzer import FrequencyAnalyzer
from ..institutional_tracking_service.tracker import InstitutionalTracker
from ...models.memory_entry import MemoryEntry, MemoryEntryType

# --- Constants ---
ORCHESTRATION_SOURCE = "OrchestrationService"

class OrchestrationDaemon:
    """
    Main control daemon for the AI Trading System.
    Initializes and coordinates all services, manages the main execution loop,
    monitors health, and handles scheduling and notifications.
    """

    def __init__(self):
        log.info("Initializing OrchestrationDaemon...")
        self._running = False
        self._optimal_trading_frequency_sec: int = config.OPTIMIZATION_MIN_FREQUENCY # Start with min
        self._last_trade_cycle_time: Optional[datetime] = None

        try:
            # --- Initialize Interfaces ---
            log.info("Initializing interfaces...")
            self.brokerage = BrokerageInterface()
            self.llm = LLMInterface()
            self.notifier = NotificationInterface()
            self.institutional_data = InstitutionalDataInterface()

            # --- Initialize Services ---
            log.info("Initializing services...")
            self.memory_storage = MemoryStorage()
            self.memory_organizer = MemoryOrganizer(self.memory_storage, self.llm)
            
            # Initialize institutional tracker
            self.institutional_tracker = InstitutionalTracker(
                institutional_data=self.institutional_data,
                llm=self.llm,
                memory_storage=self.memory_storage
            )
            
            # Initialize AI processor with institutional tracker
            self.ai_processor = AIServiceProcessor(
                llm_interface=self.llm,
                brokerage_interface=self.brokerage,
                memory_storage=self.memory_storage,
                institutional_tracker=self.institutional_tracker
            )
            
            self.execution_manager = ExecutionServiceManager(self.brokerage, self.memory_storage)
            self.frequency_analyzer = FrequencyAnalyzer(self.memory_storage)
            self.optimization_engine = OptimizationEngine(self.memory_storage, self.llm)

            log.info("OrchestrationDaemon initialized successfully.")
            self._log_system_event("System Startup")

        except TradingSystemError as e:
            log.critical(f"Initialization failed due to Trading System Error: {e}", exc_info=True)
            # Attempt to notify if possible
            try:
                self.notifier.send_notification(f"CRITICAL: System initialization failed: {e}", subject="Trading System CRITICAL FAILURE")
            except Exception as notify_err:
                log.error(f"Failed to send critical failure notification: {notify_err}")
            raise # Re-raise after attempting notification
        except Exception as e:
            log.critical(f"Unexpected critical error during initialization: {e}", exc_info=True)
            try:
                self.notifier.send_notification(f"CRITICAL: Unexpected system initialization error: {e}", subject="Trading System CRITICAL FAILURE")
            except Exception as notify_err:
                log.error(f"Failed to send critical failure notification: {notify_err}")
            raise # Re-raise

    def _log_system_event(self, event_description: str, payload: Optional[dict] = None):
        """Helper to log system-level events to memory."""
        event_payload = {"event": event_description}
        if payload:
            event_payload.update(payload)
        entry = MemoryEntry(
            entry_type=MemoryEntryType.SYSTEM_EVENT,
            source_service=ORCHESTRATION_SOURCE,
            payload=event_payload
        )
        try:
            self.memory_storage.save_memory(entry)
        except MemoryServiceError as e:
            log.error(f"Failed to save system event memory entry ('{event_description}'): {e}")

    def _handle_shutdown(self, signum, frame):
        """Gracefully handles shutdown signals (SIGINT, SIGTERM)."""
        log.warning(f"Received shutdown signal {signum}. Initiating graceful shutdown...")
        self._running = False
        # Note: schedule library doesn't automatically stop jobs on exit.
        # The loop condition `while self._running:` will handle termination.
        # Add any specific cleanup here if needed (e.g., liquidating positions).
        if config.LIQUIDATE_ON_CLOSE: # Use this config for shutdown liquidation too?
             log.warning("Attempting to liquidate positions on shutdown (if configured)...")
             # TODO: Implement liquidation logic by calling ExecutionServiceManager
             pass
        self._log_system_event("System Shutdown Initiated", {"signal": signum})

    def _run_health_checks(self):
        """Performs periodic health checks on critical components."""
        log.debug("Running periodic health checks...")
        all_ok = True
        try:
            # Check Brokerage Connection (e.g., market status or account info)
            self.brokerage.is_market_open() # Simple check
            log.debug("Brokerage connection check OK.")
        except BrokerageError as e:
            log.error(f"Health Check FAIL: Brokerage connection error: {e}")
            self.notifier.send_notification(f"HEALTH ALERT: Brokerage connection failed: {e}", subject="Trading System Health Alert")
            all_ok = False

        # Check LLM (if possible - e.g., simple test call or model list)
        # try:
        #     # self.llm.generate_response("health check", max_tokens=5) # Example test
        #     log.debug("LLM connection check OK (basic).")
        # except LLMError as e:
        #     log.error(f"Health Check FAIL: LLM error: {e}")
        #     self.notifier.send_notification(f"HEALTH ALERT: LLM API failed: {e}", subject="Trading System Health Alert")
        #     all_ok = False

        # Check Memory Service (e.g., list files in 'cur')
        try:
            self.memory_storage.list_files("cur")
            log.debug("Memory storage check OK.")
        except MemdirIOError as e:
            log.error(f"Health Check FAIL: Memory storage error: {e}")
            self.notifier.send_notification(f"HEALTH ALERT: Memory storage access failed: {e}", subject="Trading System Health Alert")
            all_ok = False

        # Check for high error rate in memory (optional)
        # ... query memory for ERROR entries ...

        if all_ok:
            log.info("Health checks passed.")
        else:
            log.warning("One or more health checks failed.")
        self._log_system_event("Health Check Run", {"status": "OK" if all_ok else "FAILED"})


    def _run_memory_organization(self):
        """Triggers the Memory Organizer to process new files."""
        log.info("Triggering memory organization task...")
        try:
            processed_count = self.memory_organizer.process_new_memories()
            log.info(f"Memory organization task finished. Processed {processed_count} files.")
            self._log_system_event("Memory Organization Run", {"processed_count": processed_count})
        except Exception as e:
            log.error(f"Error during scheduled memory organization: {e}", exc_info=True)
            self.notifier.send_notification(f"ERROR: Memory organization task failed: {e}", subject="Trading System Error")
            self._log_system_event("Memory Organization Run Failed", {"error": str(e)})

    def _run_optimization_cycle(self):
        """Triggers the Optimization Engine cycle."""
        log.info("Triggering optimization cycle task...")
        try:
            self.optimization_engine.run_optimization_cycle()
            # After optimization, recalculate frequency
            self._update_optimal_frequency()
            self._log_system_event("Optimization Cycle Run Completed")
        except Exception as e:
            log.error(f"Error during scheduled optimization cycle: {e}", exc_info=True)
            self.notifier.send_notification(f"ERROR: Optimization cycle task failed: {e}", subject="Trading System Error")
            self._log_system_event("Optimization Cycle Run Failed", {"error": str(e)})

    def _run_institutional_analysis(self):
        """Triggers institutional tracking analysis to update whale following signals."""
        log.info("Running institutional analysis (whale following)...")
        try:
            # Analyze institutional movements for our target symbols
            symbols = config.DEFAULT_SYMBOLS
            analysis_result = self.institutional_tracker.analyze_institutional_movements(symbols)
            
            # Log results
            if analysis_result.strong_signals:
                log.info(f"Institutional analysis completed: {len(analysis_result.strong_signals)} strong signals detected")
                
                # Send notification about strong institutional signals
                signal_summary = []
                for signal in analysis_result.strong_signals[:3]:  # Top 3 signals
                    signal_summary.append(f"• {signal.symbol}: {signal.action_recommendation.value} ({signal.confidence:.0%} confidence)")
                
                notification_msg = f"🐋 INSTITUTIONAL SIGNALS DETECTED:\n" + "\n".join(signal_summary)
                self.notifier.send_notification(notification_msg, subject="Whale Following Alert")
            else:
                log.info("Institutional analysis completed: No strong signals detected")
            
            self._log_system_event("Institutional Analysis Completed", {
                "total_signals": analysis_result.total_signals,
                "strong_signals": len(analysis_result.strong_signals),
                "symbols_analyzed": analysis_result.symbols_analyzed
            })
            
        except Exception as e:
            log.error(f"Error during institutional analysis: {e}", exc_info=True)
            self.notifier.send_notification(f"ERROR: Institutional analysis failed: {e}", subject="Trading System Error")
            self._log_system_event("Institutional Analysis Failed", {"error": str(e)})

    def _update_optimal_frequency(self):
        """Calculates and updates the optimal trading frequency."""
        log.info("Updating optimal trading frequency...")
        try:
            calculated_freq = self.frequency_analyzer.calculate_optimal_frequency()
            if calculated_freq is not None:
                if calculated_freq != self._optimal_trading_frequency_sec:
                     log.info(f"Optimal trading frequency updated: {self._optimal_trading_frequency_sec}s -> {calculated_freq}s")
                     self._optimal_trading_frequency_sec = calculated_freq
                     self._log_system_event("Optimal Frequency Updated", {"new_frequency_sec": calculated_freq})
                else:
                     log.info(f"Optimal trading frequency remains unchanged: {self._optimal_trading_frequency_sec}s")
            else:
                 log.warning("Failed to calculate optimal frequency. Using previous value or minimum.")
                 # Keep the old value or default to minimum if calculation fails
                 self._optimal_trading_frequency_sec = max(config.OPTIMIZATION_MIN_FREQUENCY, self._optimal_trading_frequency_sec)

        except Exception as e:
            log.error(f"Error updating optimal frequency: {e}", exc_info=True)
            # Keep using the old frequency on error

    def _run_trading_cycle(self):
        """Executes a single trading cycle: data -> AI -> execution."""
        log.info("--- Starting Trading Cycle ---")
        cycle_start_time = time.time()
        self._last_trade_cycle_time = datetime.now(timezone.utc)

        try:
            # 1. Check Market Status (crypto trades 24/7, stocks need market open)
            portfolio = self.execution_manager.get_current_portfolio(force_refresh=True) # Get portfolio first
            symbols_to_watch = set(config.DEFAULT_SYMBOLS) | set(portfolio.positions.keys())
            
            # Separate crypto and stock symbols
            crypto_symbols = {s for s in symbols_to_watch if 'USD' in s and len(s) <= 7}  # BTCUSD, ETHUSD, etc.
            stock_symbols = symbols_to_watch - crypto_symbols
            
            # Check if we can trade anything
            market_open = self.brokerage.is_market_open()
            if not market_open and not crypto_symbols:
                log.info("Trading cycle skipped: Market closed and no crypto positions.")
                return
                
            if not market_open and stock_symbols:
                log.info(f"Market closed - trading crypto only: {crypto_symbols}")
                symbols_to_watch = crypto_symbols  # Only trade crypto when market closed

            # 2. Get Market Data & Portfolio State
            log.debug("Fetching latest market data and portfolio state...")
            if not symbols_to_watch:
                log.info("No symbols to trade this cycle.")
                return
            market_data = self.brokerage.get_latest_market_data(list(symbols_to_watch))

            # 3. Trigger AI Analysis -> Signal Generation
            log.debug("Generating AI trading signal...")
            # TODO: Select appropriate prompt dynamically if needed
            signal = self.ai_processor.generate_trading_signal(market_data, portfolio)

            # 4. Process Signal -> Execution
            if signal:
                log.debug(f"Processing generated signal for {signal.symbol}...")
                submitted_order = self.execution_manager.process_signal(signal)
                if submitted_order:
                    log.info(f"Order submitted based on signal: {submitted_order.side.value} {submitted_order.qty} {submitted_order.symbol} (ID: {submitted_order.id})")
                    # Optionally notify on successful order submission
                    # self.notifier.send_notification(f"Order Submitted: {submitted_order.side.value} {submitted_order.qty} {submitted_order.symbol} (ID: {submitted_order.id})")
                else:
                    log.warning(f"Signal for {signal.symbol} processed, but no order was submitted (failed checks or execution error).")
            else:
                log.info("No actionable signal generated by AI this cycle.")

        except TradingSystemError as e:
             log.error(f"Error during trading cycle: {e}", exc_info=True)
             self.notifier.send_notification(f"ERROR in Trading Cycle: {e}", subject="Trading System Error")
        except Exception as e:
             log.critical(f"Unexpected critical error during trading cycle: {e}", exc_info=True)
             self.notifier.send_notification(f"CRITICAL ERROR in Trading Cycle: {e}", subject="Trading System CRITICAL ERROR")
        finally:
             cycle_end_time = time.time()
             duration_ms = (cycle_end_time - cycle_start_time) * 1000
             log.info(f"--- Trading Cycle Finished ({duration_ms:.2f} ms) ---")
             # Log pipeline latency metric
             metric_entry = MemoryEntry(
                  entry_type=MemoryEntryType.METRIC,
                  source_service=ORCHESTRATION_SOURCE,
                  payload={"name": "pipeline_latency_ms", "value": duration_ms}
             )
             try:
                  self.memory_storage.save_memory(metric_entry)
             except MemoryServiceError as e:
                  log.error(f"Failed to save pipeline latency metric: {e}")


    def run(self):
        """Starts the main daemon loop and schedules tasks."""
        log.info("Starting Orchestration Daemon run loop...")
        self._running = True
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        # --- Initial Frequency Calculation ---
        self._update_optimal_frequency()

        # --- Setup Scheduled Tasks ---
        log.info("Setting up scheduled tasks...")
        # Schedule health checks (e.g., every 5 minutes)
        schedule.every(5).minutes.do(self._run_health_checks)
        # Schedule memory organization (e.g., every minute)
        schedule.every(1).minute.do(self._run_memory_organization)
        # Schedule institutional tracking updates (e.g., daily at 6 AM)
        if self.institutional_tracker.is_available():
            schedule.every().day.at("06:00").do(self._run_institutional_analysis)
            log.info("Scheduled daily institutional analysis at 6:00 AM")
        else:
            log.info("Institutional tracking not available - skipping scheduled updates")
        # Schedule optimization cycle (based on config - daily, weekly, etc.)
        if config.OPTIMIZATION_ENABLED:
             schedule_time = "02:00" # Default daily at 2 AM (configure?)
             if config.OPTIMIZATION_SCHEDULE == "daily":
                  schedule.every().day.at(schedule_time).do(self._run_optimization_cycle)
             elif config.OPTIMIZATION_SCHEDULE == "weekly":
                  schedule.every().monday.at(schedule_time).do(self._run_optimization_cycle) # Example: Weekly on Monday
             # Add more schedule options (e.g., parsing cron expressions) if needed
             else:
                  log.warning(f"Unsupported optimization schedule: {config.OPTIMIZATION_SCHEDULE}. Defaulting to daily.")
                  schedule.every().day.at(schedule_time).do(self._run_optimization_cycle)
             log.info(f"Optimization cycle scheduled: {config.OPTIMIZATION_SCHEDULE} at {schedule_time}")
        else:
             log.info("Optimization cycle scheduling skipped (disabled).")

        # --- Main Loop ---
        log.info("Entering main loop...")
        self._log_system_event("Main Loop Started")
        while self._running:
            now = datetime.now(timezone.utc)

            # Check if it's time for a trading cycle based on frequency
            run_trade_cycle = False
            if self._last_trade_cycle_time is None:
                 run_trade_cycle = True # Run immediately on first start if market open
            else:
                 time_since_last_cycle = (now - self._last_trade_cycle_time).total_seconds()
                 if time_since_last_cycle >= self._optimal_trading_frequency_sec:
                      run_trade_cycle = True

            if run_trade_cycle:
                 # Run the trading cycle (includes market open check inside)
                 self._run_trading_cycle()
                 # Note: _run_trading_cycle updates _last_trade_cycle_time internally on start

            # Run pending scheduled tasks
            schedule.run_pending()

            # Sleep interval - adjust dynamically?
            # Sleep for a short duration to avoid busy-waiting
            # Calculate sleep time based on next scheduled job and desired frequency check interval
            idle_time = schedule.idle_seconds()
            sleep_interval = config.MAIN_LOOP_SLEEP_INTERVAL # Base sleep
            if idle_time is not None and idle_time > 0:
                 # Sleep until the next job, but no more than the base interval
                 # to allow checking the trading frequency condition reasonably often.
                 sleep_interval = min(sleep_interval, idle_time)

            # Ensure sleep interval is positive
            sleep_interval = max(0.1, sleep_interval)

            # log.debug(f"Sleeping for {sleep_interval:.2f} seconds...")
            time.sleep(sleep_interval)

        log.info("Orchestration Daemon run loop finished.")
        self._log_system_event("Main Loop Stopped")
        schedule.clear() # Clear scheduled jobs on exit
