"""
╔══════════════════════════════════════════════════════════════════╗
║           QuantBot v7.2 — Final Fix                              ║
║                                                                  ║
║  إصلاح v7.2:                                                     ║
║  ✅ SHAPE = (3,3,3,3,3) — البُعد الخامس للـ action               ║
║  ✅ learn() يستخدم [state][action] بشكل صحيح                     ║
║  ✅ حذف q_table.npy تلقائياً عند تعارض الأبعاد                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

import ccxt.pro as ccxt
import asyncio
import random
import logging
import os
import sys
import signal
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
from datetime import datetime, date

# ─── إصلاح Windows Event Loop ────────────────────────────────────────────────
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ─── إعداد السجلات ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("QuantBot")

# ─── ثوابت المشروع ────────────────────────────────────────────────────────────
os.makedirs("memory", exist_ok=True)
Q_TABLE_FILE = "memory/q_table.npy"
TRADES_FILE  = "memory/trades_history.csv"
SYMBOL       = "SOL/USDT"

# ─── هياكل البيانات ──────────────────────────────────────────────────────────
@dataclass
class Tick:
    price:  float
    amount: float
    side:   str

@dataclass
class TradeRecord:
    timestamp:   str
    position:    str
    entry_price: float
    exit_price:  float
    size_usd:    float
    pnl_usd:     float
    pnl_pct:     float
    reason:      str

@dataclass
class BrainStep:
    state:  Tuple
    action: int


# ══════════════════════════════════════════════════════════════════════════════
# 1. RiskManager
# ══════════════════════════════════════════════════════════════════════════════
class RiskManager:
    def __init__(
        self,
        total_balance:      float = 1_000.0,
        max_daily_loss_pct: float = 0.02,
        kelly_fraction:     float = 0.5,
        min_risk_pct:       float = 0.01,
        max_risk_pct:       float = 0.05,
        reward_risk_ratio:  float = 1.5,
    ):
        self.total_balance     = total_balance
        self.max_daily_loss    = total_balance * max_daily_loss_pct
        self.kelly_fraction    = kelly_fraction
        self.min_risk_pct      = min_risk_pct
        self.max_risk_pct      = max_risk_pct
        self.reward_risk_ratio = reward_risk_ratio
        self.daily_pnl         = 0.0
        self._last_reset: date = datetime.now().date()

    def _auto_reset(self) -> None:
        today = datetime.now().date()
        if today > self._last_reset:
            log.info(f"📅 يوم جديد — تصفير PnL اليومي (كان: ${self.daily_pnl:+.2f})")
            self.daily_pnl   = 0.0
            self._last_reset = today

    def is_trading_allowed(self) -> bool:
        self._auto_reset()
        if self.daily_pnl <= -self.max_daily_loss:
            log.critical(
                f"🛑 قاطع التيار! خسارة اليوم ${self.daily_pnl:.2f} "
                f"تجاوزت الحد ${-self.max_daily_loss:.2f}"
            )
            return False
        return True

    def _get_win_rate(self, prior_wins: int = 3, prior_total: int = 6) -> float:
        try:
            df = pd.read_csv(TRADES_FILE)
            if df.empty or "pnl_usd" not in df.columns:
                return prior_wins / prior_total
            wins  = len(df[df["pnl_usd"] > 0])
            total = len(df)
            return (wins + prior_wins) / (total + prior_total)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            return prior_wins / prior_total

    def calculate_position_size(self) -> float:
        win_rate = self._get_win_rate()
        b        = self.reward_risk_ratio
        kelly    = (win_rate * b - (1 - win_rate)) / b
        risk_pct = min(max(kelly * self.kelly_fraction, self.min_risk_pct), self.max_risk_pct)
        size     = self.total_balance * risk_pct
        log.debug(f"📐 WinRate={win_rate:.2%} Kelly={kelly:.3f} Size=${size:.2f}")
        return size

    def record_pnl(self, pnl_usd: float) -> None:
        self.daily_pnl += pnl_usd


# ══════════════════════════════════════════════════════════════════════════════
# 2. MarketAnalyzer
# ══════════════════════════════════════════════════════════════════════════════
class MarketAnalyzer:
    def __init__(self, window: int = 500, atr_window: int = 100):
        self.ticks:  deque = deque(maxlen=window)
        self.prices: deque = deque(maxlen=atr_window)
        self.cvd:    float = 0.0

    def add_tick(self, tick: Tick) -> None:
        self.ticks.append(tick)
        self.prices.append(tick.price)
        self.cvd += tick.amount if tick.side == "buy" else -tick.amount

    @property
    def is_ready(self) -> bool:
        return len(self.ticks) >= 200

    def get_ofi(self) -> float:
        buys  = sum(t.amount for t in self.ticks if t.side == "buy")
        sells = sum(t.amount for t in self.ticks if t.side == "sell")
        total = buys + sells
        return (buys - sells) / total if total > 0 else 0.0

    def get_volatility(self) -> float:
        if len(self.prices) < 2:
            return 0.001
        last_price = self.prices[-1]
        return float(np.std(list(self.prices)) / last_price) if last_price > 0 else 0.001

    def get_atr(self) -> float:
        prices = list(self.prices)
        if len(prices) < 2:
            return 0.002
        diffs = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
        return float(np.mean(diffs)) / prices[-1] if prices[-1] > 0 else 0.002

    def get_cvd_signal(self) -> int:
        if self.cvd < -500: return 0
        if self.cvd >  500: return 2
        return 1


# ══════════════════════════════════════════════════════════════════════════════
# 3. QuantBrain
# ══════════════════════════════════════════════════════════════════════════════
class QuantBrain:
    # 4 أبعاد للحالة × 3 قرارات = شكل صحيح للوصول بـ [state][action]
    SHAPE = (3, 3, 3, 3, 3)

    def __init__(
        self,
        learning_rate:   float = 0.05,
        discount_factor: float = 0.9,
        explore_rate:    float = 0.1,
    ):
        self.lr      = learning_rate
        self.gamma   = discount_factor
        self.epsilon = explore_rate
        self.q_table = self._load_or_init()
        self.trajectory: List[BrainStep] = []

    def _load_or_init(self) -> np.ndarray:
        if os.path.exists(Q_TABLE_FILE):
            try:
                q = np.load(Q_TABLE_FILE)
                if q.shape == self.SHAPE:
                    log.info("🧠 استُعيدت الذاكرة من الجلسة السابقة.")
                    return q
                else:
                    log.warning(
                        f"⚠️ شكل Q-table قديم {q.shape} ≠ {self.SHAPE} "
                        f"— حذف تلقائي والبدء من الصفر."
                    )
                    os.remove(Q_TABLE_FILE)
            except Exception as e:
                log.warning(f"⚠️ تعذّر تحميل الذاكرة: {e} — بدء من الصفر.")
        log.info("🧠 بدء بعقل جديد من الصفر.")
        return np.zeros(self.SHAPE)

    def save(self) -> None:
        np.save(Q_TABLE_FILE, self.q_table)

    def _classify(self, value: float, low: float, high: float) -> int:
        if value < low:  return 0
        if value > high: return 2
        return 1

    def get_state(self, ofi: float, vol: float, cvd_signal: int) -> Tuple:
        s_ofi   = self._classify(ofi, -0.2,   0.2)
        s_vol   = self._classify(vol,  0.0005, 0.0015)
        s_cvd   = cvd_signal
        s_align = 1 if s_ofi == s_cvd else (0 if s_ofi < s_cvd else 2)
        return (s_ofi, s_vol, s_cvd, s_align)

    def select_action(self, state: Tuple) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, 2)
        return int(np.argmax(self.q_table[state]))

    def record_step(self, state: Tuple, action: int) -> None:
        self.trajectory.append(BrainStep(state=state, action=action))

    def learn(self, final_pnl_pct: float) -> None:
        if not self.trajectory:
            return

        base_reward = final_pnl_pct * 100

        for i, step in enumerate(reversed(self.trajectory)):
            # ✅ صحيح: [state] يُعيد مصفوفة 3 عناصر، [action] يختار منها
            current_q   = self.q_table[step.state][step.action]
            best_next_q = np.max(self.q_table[step.state])
            td_error    = base_reward * (self.gamma ** i) + self.gamma * best_next_q - current_q
            self.q_table[step.state][step.action] += self.lr * td_error

        log.debug(f"🎓 تعلّم من {len(self.trajectory)} خطوة | Reward={base_reward:.2f}")
        self.trajectory.clear()
        self.save()


# ══════════════════════════════════════════════════════════════════════════════
# 4. ExecutionEngine
# ══════════════════════════════════════════════════════════════════════════════
class ExecutionEngine:
    def __init__(self, brain: QuantBrain, risk: RiskManager, atr_multiplier: float = 2.0):
        self.brain    = brain
        self.risk     = risk
        self.atr_m    = atr_multiplier
        self.position:    Optional[str] = None
        self.entry_price: float         = 0.0
        self.size_usd:    float         = 0.0
        self.tp_pct:      float         = 0.0
        self.sl_pct:      float         = 0.0

    def has_position(self) -> bool:
        return self.position is not None

    def open(self, side: str, price: float, size: float, atr: float) -> None:
        if not self.risk.is_trading_allowed():
            return
        self.position    = side
        self.entry_price = price
        self.size_usd    = size
        self.sl_pct = max(atr * self.atr_m,       0.002)
        self.tp_pct = max(atr * self.atr_m * 1.5, 0.003)
        log.info(
            f"{'🟢' if side == 'LONG' else '🔴'} دخول {side} "
            f"@ {price:.3f} | TP={self.tp_pct*100:.2f}% SL={self.sl_pct*100:.2f}% "
            f"| حجم=${size:.2f}"
        )

    def monitor(self, price: float, state: Tuple, action: int) -> bool:
        if not self.position:
            return False

        # ✅ تسجيل كل خطوة للتعلم
        self.brain.record_step(state, action)

        pnl_pct = (price - self.entry_price) / self.entry_price
        if self.position == "SHORT":
            pnl_pct = -pnl_pct

        hit_tp   = pnl_pct >=  self.tp_pct
        hit_sl   = pnl_pct <= -self.sl_pct
        reversal = (
            (self.position == "LONG"  and action == 2) or
            (self.position == "SHORT" and action == 0)
        )

        if not (hit_tp or hit_sl or reversal):
            return False

        reason  = "🎯 هدف" if hit_tp else ("🛑 وقف" if hit_sl else "🔄 انعكاس")
        pnl_usd = self.size_usd * pnl_pct

        log.info(
            f"⚙️  إغلاق {self.position} @ {price:.3f} "
            f"| PnL=${pnl_usd:+.2f} ({pnl_pct*100:+.3f}%) | {reason}"
        )

        self.brain.learn(pnl_pct)
        self.risk.record_pnl(pnl_usd)
        self._save_trade(price, pnl_pct, pnl_usd, reason)

        self.position = None
        return True

    def _save_trade(self, exit_price: float, pnl_pct: float, pnl_usd: float, reason: str) -> None:
        record = TradeRecord(
            timestamp   = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            position    = self.position or "?",
            entry_price = self.entry_price,
            exit_price  = exit_price,
            size_usd    = self.size_usd,
            pnl_usd     = pnl_usd,
            pnl_pct     = pnl_pct * 100,
            reason      = reason.split()[-1],
        )
        df = pd.DataFrame([vars(record)])
        df.to_csv(
            TRADES_FILE,
            mode   = "a",
            header = not os.path.exists(TRADES_FILE),
            index  = False,
        )


# ══════════════════════════════════════════════════════════════════════════════
# 5. LiveMarketStreamer
# ══════════════════════════════════════════════════════════════════════════════
class LiveMarketStreamer:
    def __init__(self, symbol: str = SYMBOL):
        self.symbol   = symbol
        self.exchange = ccxt.bybit({
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })
        self.analyzer  = MarketAnalyzer()
        self.brain     = QuantBrain()
        self.risk      = RiskManager()
        self.execution = ExecutionEngine(self.brain, self.risk)
        self._running  = True

    async def stream(self) -> None:
        await self.exchange.load_markets()
        log.info(f"✅ متصل بـ Bybit | الزوج: {self.symbol} | v7.2 جاهز")

        while self._running:
            try:
                raw_trades = await self.exchange.watch_trades(self.symbol)

                for t in raw_trades:
                    price  = t.get("price")
                    amount = t.get("amount")
                    side   = t.get("side")

                    if not all([price, amount, side]):
                        continue

                    tick = Tick(price=price, amount=amount, side=side)
                    self.analyzer.add_tick(tick)

                    if not self.analyzer.is_ready:
                        continue

                    ofi        = self.analyzer.get_ofi()
                    vol        = self.analyzer.get_volatility()
                    cvd_signal = self.analyzer.get_cvd_signal()
                    atr        = self.analyzer.get_atr()

                    state  = self.brain.get_state(ofi, vol, cvd_signal)
                    action = self.brain.select_action(state)

                    if self.execution.has_position():
                        self.execution.monitor(price, state, action)
                    elif action != 1 and self.risk.is_trading_allowed():
                        self.brain.record_step(state, action)
                        size = self.risk.calculate_position_size()
                        side_str = "LONG" if action == 0 else "SHORT"
                        self.execution.open(side_str, price, size, atr)

            except ccxt.NetworkError as e:
                log.warning(f"🌐 انقطاع شبكة — إعادة اتصال... ({e})")
                await asyncio.sleep(2)
            except ccxt.ExchangeError as e:
                log.error(f"⚠️ خطأ منصة: {e}")
                await asyncio.sleep(5)
            except Exception as e:
                log.exception(f"❌ خطأ غير متوقع: {e}")
                break

    async def shutdown(self) -> None:
        self._running = False
        self.brain.save()
        await self.exchange.close()
        log.info("👋 تم إغلاق الاتصال وحفظ الذاكرة بأمان.")


# ══════════════════════════════════════════════════════════════════════════════
# 6. نقطة الدخول
# ══════════════════════════════════════════════════════════════════════════════
async def main() -> None:
    bot  = LiveMarketStreamer(SYMBOL)
    loop = asyncio.get_event_loop()

    def _handle_signal():
        log.info("🛑 إشارة إيقاف مستلمة...")
        asyncio.ensure_future(bot.shutdown())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_signal)
        except NotImplementedError:
            pass

    try:
        await bot.stream()
    except KeyboardInterrupt:
        log.info("🛑 إيقاف يدوي.")
    finally:
        await bot.shutdown()


if __name__ == "__main__":
    asyncio.run(main())