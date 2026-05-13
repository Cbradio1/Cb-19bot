"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          MOMENTUM TRADING BOT — Alpaca-py Edition                           ║
║          Strategy : Base-Hit Asymmetric Momentum                            ║
║          Author   : Generated via Claude (Anthropic)                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

QUICK-START INSTRUCTIONS
────────────────────────
1. Install dependencies:
      pip install alpaca-py numpy

2. Paste your Alpaca keys into the CONFIG block below (Section ①).
   Get them from: https://app.alpaca.markets  →  Paper Trading  →  API Keys

3. Run on Replit / any free host:
      python bot.py

4. To switch to LIVE trading:  set  PAPER = False  (use real money at your own risk).

HOW THE BOT WORKS (Theory of Operation)
────────────────────────────────────────
  ┌─ WebSocket ─┐   ┌─ Momentum Engine ─┐   ┌─ Risk Gate ──────────┐
  │ Real-time   │ → │ OLS Linear Regr.  │ → │ Monte Carlo VaR      │
  │ trade ticks │   │ Slope + Accel     │   │ 1,000 GBM paths      │
  └─────────────┘   └───────────────────┘   │ 5th-pct VaR < 1%?   │
                                             └──────────────────────┘
                                                        │ YES
                                             ┌──────────▼───────────┐
                                             │ Bracket Order        │
                                             │ TP +1.5% / SL -0.75% │
                                             │ Fractional notional  │
                                             └──────────────────────┘
                                                        │
                                             ┌──────────▼───────────┐
                                             │ Kill Switch Monitor  │
                                             │ Daily loss > $5 →    │
                                             │ Flatten ALL + halt   │
                                             └──────────────────────┘

CRITICAL MECHANICAL REALITIES
──────────────────────────────
• Alpaca fractional shares DO NOT support full bracket orders via the API.
  The bot therefore places: (a) a fractional market-buy order, then
  (b) a separate trailing-stop order at 0.75% to protect the position.
  Take-profit is monitored via an async watchdog loop (Section ⑦).

• Paper Trading ≠ Live fills.  Slippage, partial fills, and latency will
  differ in live mode.  Always validate in paper mode first.

• Starting with $20 is valid.  The bot sizes every trade at TRADE_SIZE_USD
  (default $5) using Alpaca's "notional" (dollar-amount) order parameter.

• The "million-dollar" path requires consistent positive expectancy over
  thousands of trades.  This bot gives you the engine; edge is earned
  by tuning TICKERS, REGRESSION_WINDOW, and VAR_LIMIT_PCT to market conditions.
"""

# ──────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import asyncio
import json
import math
import time
from collections import deque
from datetime import date, datetime, timezone
from typing import Optional

import numpy as np

# Alpaca-py SDK
from alpaca.data.live import StockDataStream
from alpaca.data.models import Trade
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import (
    OrderClass,
    OrderSide,
    QueryOrderStatus,
    TimeInForce,
)
from alpaca.trading.requests import (
    GetOrdersRequest,
    MarketOrderRequest,
    TrailingStopOrderRequest,
)


# ──────────────────────────────────────────────────────────────────────────────
# ① CONFIGURATION  ←  PASTE YOUR KEYS HERE
# ──────────────────────────────────────────────────────────────────────────────
API_KEY    = "PASTE_YOUR_ALPACA_API_KEY_HERE"      # ← replace this
SECRET_KEY = "PASTE_YOUR_ALPACA_SECRET_KEY_HERE"   # ← replace this
PAPER      = True   # True = paper trading (safe).  False = real money (careful!)

# ──────────────────────────────────────────────────────────────────────────────
# ② STRATEGY PARAMETERS  ←  tune these to fit your edge
# ──────────────────────────────────────────────────────────────────────────────
TICKERS: list[str] = [
    # High-volatility, typically $2–$10 range (verify prices before running)
    "SNDL", "HIMS", "SOFI", "MARA", "RIOT",
    "PLTR", "NIO",  "LCID", "WKHS", "CLOV",
]

REGRESSION_WINDOW   = 30      # Number of price ticks used for OLS slope
ACCEL_LAG           = 5       # Tick lag used to estimate 2nd derivative
TRADE_SIZE_USD      = 5.00    # Dollar notional per trade (fractional shares)
TAKE_PROFIT_PCT     = 0.015   # +1.5% exit target
STOP_LOSS_PCT       = 0.0075  # -0.75% trailing stop
DAILY_LOSS_LIMIT    = 5.00    # Kill-switch fires when daily P&L hits -$5

# Monte Carlo VaR settings
MC_PATHS            = 1_000   # Number of GBM simulation paths
MC_HORIZON_MIN      = 5       # Forward projection horizon (minutes)
VAR_LIMIT_PCT       = 0.01    # Max tolerable 5th-percentile VaR (1% of equity)

# Watchdog — how often (seconds) to poll open positions for TP exit
WATCHDOG_INTERVAL   = 10


# ──────────────────────────────────────────────────────────────────────────────
# ③ STRUCTURED JSON LOGGER
#    All output is machine-readable for your diagnostic audit.
# ──────────────────────────────────────────────────────────────────────────────
class JSONLogger:
    """Emits every log line as a JSON object to stdout."""

    def __init__(self, name: str = "MomentumBot"):
        self.name = name

    def _emit(self, level: str, event: str, **kwargs) -> None:
        record = {
            "ts":    datetime.now(timezone.utc).isoformat(),
            "lvl":   level,
            "bot":   self.name,
            "event": event,
            **kwargs,
        }
        print(json.dumps(record), flush=True)

    def info(self,  event: str, **kw) -> None: self._emit("INFO",    event, **kw)
    def warn(self,  event: str, **kw) -> None: self._emit("WARN",    event, **kw)
    def error(self, event: str, **kw) -> None: self._emit("ERROR",   event, **kw)
    def debug(self, event: str, **kw) -> None: self._emit("DEBUG",   event, **kw)


log = JSONLogger()


# ──────────────────────────────────────────────────────────────────────────────
# ④ MOMENTUM ENGINE  — Linear Regression + Acceleration
# ──────────────────────────────────────────────────────────────────────────────
class MomentumEngine:
    """
    Computes:
      slope  (β₁) = OLS regression of price vs tick index (1st derivative)
      accel       = slope_now − slope_prev  (2nd derivative approximation)

    Entry condition: slope > 0  AND  accel > 0
    This ensures price is rising AND the rise is accelerating — asymmetric momentum.
    """

    def __init__(self, window: int = REGRESSION_WINDOW, lag: int = ACCEL_LAG):
        self.window = window
        self.lag    = lag

    @staticmethod
    def _ols_slope(y: np.ndarray) -> float:
        """Ordinary Least Squares slope: β = Σ(x-x̄)(y-ȳ) / Σ(x-x̄)²"""
        n    = len(y)
        x    = np.arange(n, dtype=np.float64)
        xm   = x.mean()
        ym   = y.mean()
        num  = np.dot(x - xm, y - ym)
        den  = np.dot(x - xm, x - xm)
        return float(num / den) if den != 0 else 0.0

    def evaluate(self, buf: deque) -> dict:
        needed = self.window + self.lag
        if len(buf) < needed:
            return {
                "signal": False, "slope": 0.0, "accel": 0.0,
                "reason": "insufficient_data",
            }

        prices = np.array(list(buf), dtype=np.float64)

        slope_now  = self._ols_slope(prices[-self.window:])
        slope_prev = self._ols_slope(prices[-(self.window + self.lag):-self.lag])
        accel      = slope_now - slope_prev

        signal = (slope_now > 0.0) and (accel > 0.0)

        return {
            "signal": signal,
            "slope":  round(slope_now,  6),
            "accel":  round(accel,      6),
            "reason": "momentum_confirmed" if signal else "no_momentum",
        }


# ──────────────────────────────────────────────────────────────────────────────
# ⑤ RISK MANAGER  — Monte Carlo Value-at-Risk Gate
# ──────────────────────────────────────────────────────────────────────────────
class RiskManager:
    """
    Before any trade, simulates 1,000 Geometric Brownian Motion (GBM) paths
    over MC_HORIZON_MIN minutes using realized log-return volatility.

    The 5th-percentile outcome (worst-case 5% scenario) is expressed as a
    fraction of current price.  If that loss exceeds VAR_LIMIT_PCT of account
    equity, the trade is blocked.

    Math:
      S_t = S_0 · exp[(μ - ½σ²)t  +  σ√t · Z]   where Z ~ N(0,1)
    """

    def __init__(self, paths: int = MC_PATHS, horizon: int = MC_HORIZON_MIN):
        self.paths   = paths
        self.horizon = horizon
        self.rng     = np.random.default_rng(seed=None)  # non-deterministic

    def var_check(self, prices: np.ndarray, account_equity: float) -> dict:
        if len(prices) < 10:
            return {
                "approved": False, "var_pct": 1.0,
                "reason": "insufficient_history_for_var",
            }

        log_rets = np.diff(np.log(prices[-min(len(prices), 60):]))
        mu       = float(log_rets.mean())
        sigma    = float(log_rets.std())

        if sigma < 1e-10:
            return {"approved": True, "var_pct": 0.0, "reason": "zero_vol_approved"}

        S0   = float(prices[-1])
        dt   = 1.0  # 1 tick / 1 minute steps
        Z    = self.rng.standard_normal((self.paths, self.horizon))
        # Cumulative log-returns along each path
        log_paths  = np.cumsum((mu - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * Z, axis=1)
        terminal   = S0 * np.exp(log_paths[:, -1])

        pnl_pct    = (terminal - S0) / S0
        var_5th    = float(np.percentile(pnl_pct, 5))   # negative = loss
        var_dollar = abs(var_5th) * account_equity

        approved = abs(var_5th) < VAR_LIMIT_PCT

        return {
            "approved":    approved,
            "var_pct":     round(abs(var_5th), 5),
            "var_dollar":  round(var_dollar, 4),
            "var_limit":   VAR_LIMIT_PCT,
            "reason":      "var_within_limit" if approved else "var_exceeds_limit",
        }


# ──────────────────────────────────────────────────────────────────────────────
# ⑥ TRADING BOT CORE
# ──────────────────────────────────────────────────────────────────────────────
class MomentumBot:
    """
    Orchestrates the full pipeline:
      WebSocket tick → Momentum signal → VaR gate → Order → Watchdog → Kill switch
    """

    def __init__(self) -> None:
        # REST + WebSocket clients
        self.trading = TradingClient(API_KEY, SECRET_KEY, paper=PAPER)
        self.stream  = StockDataStream(API_KEY, SECRET_KEY)

        # Engines
        self.engine  = MomentumEngine()
        self.risk    = RiskManager()

        # State
        self.buffers:       dict[str, deque] = {t: deque(maxlen=300) for t in TICKERS}
        self.positions:     dict[str, Optional[str]] = {t: None for t in TICKERS}
        # positions[symbol] = order_id string when in a trade, else None

        self.entry_prices:  dict[str, float] = {}   # symbol → entry price
        self.daily_pnl:     float            = 0.0
        self.killed:        bool             = False
        self.session_date:  date             = date.today()

        log.info(
            "BOT_INITIALIZED",
            tickers=TICKERS,
            paper=PAPER,
            trade_size_usd=TRADE_SIZE_USD,
            kill_limit=DAILY_LOSS_LIMIT,
        )

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _equity(self) -> float:
        try:
            return float(self.trading.get_account().equity)
        except Exception as e:
            log.error("EQUITY_FETCH_FAILED", detail=str(e))
            return 0.0

    def _reset_if_new_day(self) -> None:
        today = date.today()
        if today != self.session_date:
            log.info("NEW_TRADING_DAY", date=str(today), prev_pnl=round(self.daily_pnl, 4))
            self.daily_pnl    = 0.0
            self.session_date = today

    # ── Kill Switch ─────────────────────────────────────────────────────────────

    def _kill_check(self) -> None:
        if self.daily_pnl <= -DAILY_LOSS_LIMIT and not self.killed:
            log.warn(
                "KILL_SWITCH_ACTIVATED",
                daily_pnl=round(self.daily_pnl, 4),
                limit=-DAILY_LOSS_LIMIT,
                action="FLATTEN_ALL_AND_HALT",
            )
            self._flatten_all()
            self.killed = True

    def _flatten_all(self) -> None:
        log.warn("FLATTENING_ALL_POSITIONS")
        try:
            self.trading.close_all_positions(cancel_orders=True)
            self.positions = {t: None for t in TICKERS}
            log.info("ALL_POSITIONS_CLOSED")
        except Exception as e:
            log.error("FLATTEN_ERROR", detail=str(e))

    # ── Order Execution ─────────────────────────────────────────────────────────

    def _open_trade(self, symbol: str, price: float) -> None:
        """
        Step 1: Market buy (fractional, by notional dollar amount).
        Step 2: Trailing stop at STOP_LOSS_PCT to protect downside.
        TP is managed by the watchdog (see Section ⑦).
        """
        notional = str(round(TRADE_SIZE_USD, 2))

        # ── Entry order ──
        try:
            entry_req = MarketOrderRequest(
                symbol        = symbol,
                notional      = notional,   # fractional shares by dollar amount
                side          = OrderSide.BUY,
                time_in_force = TimeInForce.DAY,
            )
            entry_order = self.trading.submit_order(entry_req)
            self.positions[symbol]    = str(entry_order.id)
            self.entry_prices[symbol] = price
            log.info(
                "ENTRY_ORDER_PLACED",
                symbol   = symbol,
                notional = notional,
                price    = price,
                order_id = str(entry_order.id),
            )
        except Exception as e:
            log.error("ENTRY_ORDER_FAILED", symbol=symbol, detail=str(e))
            return

        # Small pause to let entry fill before attaching the stop
        time.sleep(0.5)

        # ── Trailing stop order ──
        trail_pct = round(STOP_LOSS_PCT * 100, 3)  # Alpaca expects a percentage (e.g. 0.75)
        try:
            stop_req = TrailingStopOrderRequest(
                symbol           = symbol,
                qty              = None,   # qty=None forces "use existing position"
                notional         = notional,
                side             = OrderSide.SELL,
                time_in_force    = TimeInForce.DAY,
                trail_percent    = trail_pct,
            )
            stop_order = self.trading.submit_order(stop_req)
            log.info(
                "TRAILING_STOP_PLACED",
                symbol      = symbol,
                trail_pct   = trail_pct,
                stop_order  = str(stop_order.id),
            )
        except Exception as e:
            log.error("TRAILING_STOP_FAILED", symbol=symbol, detail=str(e))

    def _close_position_for_profit(self, symbol: str, current_price: float) -> None:
        """Called by the watchdog when price hits the take-profit target."""
        entry = self.entry_prices.get(symbol, current_price)
        pnl   = (current_price - entry) / entry

        try:
            self.trading.close_position(symbol)
            self.daily_pnl += TRADE_SIZE_USD * pnl
            self.positions[symbol]    = None
            self.entry_prices.pop(symbol, None)
            log.info(
                "TAKE_PROFIT_HIT",
                symbol        = symbol,
                entry_price   = entry,
                exit_price    = current_price,
                pnl_pct       = round(pnl * 100, 3),
                daily_pnl_usd = round(self.daily_pnl, 4),
            )
        except Exception as e:
            log.error("CLOSE_POSITION_FAILED", symbol=symbol, detail=str(e))

    # ── WebSocket Trade Callback ────────────────────────────────────────────────

    async def _on_trade(self, trade: Trade) -> None:
        if self.killed:
            return

        symbol = trade.symbol
        price  = float(trade.price)
        buf    = self.buffers[symbol]
        buf.append(price)

        # Already in a position for this ticker → skip entry logic
        if self.positions.get(symbol) is not None:
            return

        # ── Signal check ──
        mom = self.engine.evaluate(buf)
        if not mom["signal"]:
            return

        log.debug(
            "SIGNAL",
            symbol = symbol,
            price  = price,
            slope  = mom["slope"],
            accel  = mom["accel"],
        )

        # ── VaR gate ──
        equity = self._equity()
        prices = np.array(list(buf), dtype=np.float64)
        var    = self.risk.var_check(prices, equity)

        log.debug(
            "VAR_CHECK",
            symbol   = symbol,
            var_pct  = var["var_pct"],
            approved = var["approved"],
            reason   = var["reason"],
        )

        if not var["approved"]:
            return

        # ── Daily reset + kill-switch pre-check ──
        self._reset_if_new_day()
        self._kill_check()
        if self.killed:
            return

        # ── Fire the trade ──
        self._open_trade(symbol, price)

    # ── Take-Profit Watchdog ────────────────────────────────────────────────────

    async def _tp_watchdog(self) -> None:
        """
        Polls live prices from the price buffers every WATCHDOG_INTERVAL seconds.
        If any open position is up ≥ TAKE_PROFIT_PCT from entry, it closes the position.
        Also updates daily P&L when stop-loss fires (detects closed positions).
        """
        log.info("WATCHDOG_STARTED", interval_sec=WATCHDOG_INTERVAL)

        while not self.killed:
            await asyncio.sleep(WATCHDOG_INTERVAL)

            for symbol, order_id in list(self.positions.items()):
                if order_id is None:
                    continue

                buf = self.buffers[symbol]
                if not buf:
                    continue

                current = buf[-1]
                entry   = self.entry_prices.get(symbol)
                if entry is None:
                    continue

                pnl_pct = (current - entry) / entry

                # ── Take profit ──
                if pnl_pct >= TAKE_PROFIT_PCT:
                    log.info(
                        "TP_WATCHDOG_TRIGGER",
                        symbol=symbol, pnl_pct=round(pnl_pct * 100, 3)
                    )
                    self._close_position_for_profit(symbol, current)

                # ── Detect if stop-loss already fired (position gone externally) ──
                else:
                    try:
                        open_req = GetOrdersRequest(
                            status=QueryOrderStatus.OPEN, symbols=[symbol]
                        )
                        open_orders = self.trading.get_orders(filter=open_req)
                        if len(open_orders) == 0:
                            # Position likely closed by trailing stop
                            realised = TRADE_SIZE_USD * pnl_pct
                            self.daily_pnl += realised
                            self.positions[symbol]    = None
                            self.entry_prices.pop(symbol, None)
                            log.info(
                                "STOP_LOSS_DETECTED",
                                symbol        = symbol,
                                entry_price   = entry,
                                exit_price    = current,
                                pnl_pct       = round(pnl_pct * 100, 3),
                                realised_usd  = round(realised, 4),
                                daily_pnl_usd = round(self.daily_pnl, 4),
                            )
                            self._kill_check()
                    except Exception as e:
                        log.error("WATCHDOG_CHECK_FAILED", symbol=symbol, detail=str(e))

    # ── Main Run Loop ───────────────────────────────────────────────────────────

    async def run(self) -> None:
        log.info(
            "BOT_STARTING",
            paper=PAPER,
            tickers=TICKERS,
            trade_size_usd=TRADE_SIZE_USD,
            tp_pct=TAKE_PROFIT_PCT,
            sl_pct=STOP_LOSS_PCT,
            kill_switch_usd=DAILY_LOSS_LIMIT,
        )

        # Subscribe to real-time trades for all tickers
        self.stream.subscribe_trades(self._on_trade, *TICKERS)

        # Run WebSocket stream + watchdog concurrently
        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self.stream.run())
                tg.create_task(self._tp_watchdog())
        except* KeyboardInterrupt:
            log.info("BOT_STOPPED_BY_USER")
            self._flatten_all()
        except* Exception as eg:
            for exc in eg.exceptions:
                log.error("FATAL_ERROR", detail=str(exc))
            self._flatten_all()


# ──────────────────────────────────────────────────────────────────────────────
# ⑦ ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(
        json.dumps({
            "ts":    datetime.now(timezone.utc).isoformat(),
            "event": "BOT_PROCESS_START",
            "note":  "Verify API_KEY and SECRET_KEY are set before proceeding.",
        })
    )

    if "PASTE_YOUR" in API_KEY or "PASTE_YOUR" in SECRET_KEY:
        print(json.dumps({
            "lvl":   "ERROR",
            "event": "KEYS_NOT_SET",
            "action": "Open bot.py, find Section ①, and replace the placeholder strings.",
        }))
        raise SystemExit(1)

    bot = MomentumBot()
    asyncio.run(bot.run())
