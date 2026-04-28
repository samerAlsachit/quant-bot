import ccxt.pro as ccxt
import asyncio
import logging
import os
import sys
from collections import deque
import numpy as np
import pandas as pd
from datetime import datetime

# 🟢 الحل النهائي لمشكلة إغلاق الجلسات في ويندوز
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# إعداد السجلات
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("QuantBot")

# إعداد مجلد الذاكرة وملفاتها
os.makedirs("memory", exist_ok=True)
Q_TABLE_FILE = "memory/q_table.npy"
TRADES_FILE = "memory/trades_history.csv"

class RiskManager:
    def __init__(self, total_balance=1000.0, kelly_fraction=0.5):
        self.total_balance = total_balance
        self.kelly_fraction = kelly_fraction

    def calculate_position_size(self, win_rate: float, reward_risk_ratio: float) -> float:
        if win_rate <= 0 or reward_risk_ratio <= 0: return 0.0, 0.0
        kelly_pct = (win_rate * reward_risk_ratio - (1 - win_rate)) / reward_risk_ratio
        if kelly_pct <= 0: return 0.0, 0.0
        optimal_risk_pct = min(kelly_pct * self.kelly_fraction, 0.05)
        return self.total_balance * optimal_risk_pct, optimal_risk_pct

class OrderFlowManager:
    def __init__(self, window_size=50):
        self.ticks = deque(maxlen=window_size)
        self.cumulative_delta = 0.0

    def process_tick(self, price: float, amount: float, side: str):
        self.cumulative_delta += amount if side.upper() == 'BUY' else -amount
        self.ticks.append({'price': price, 'amount': amount, 'side': side.upper()})
        buy_vol = sum(t['amount'] for t in self.ticks if t['side'] == 'BUY')
        sell_vol = sum(t['amount'] for t in self.ticks if t['side'] == 'SELL')
        total_vol = buy_vol + sell_vol
        ofi = (buy_vol - sell_vol) / total_vol if total_vol > 0 else 0.0
        return ofi, self.cumulative_delta

class QuantBrain:
    def __init__(self):
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.exploration_rate = 0.1 
        self.last_state = None
        self.last_action = None
        self.last_price = None
        self._load_memory()

    def _load_memory(self):
        if os.path.exists(Q_TABLE_FILE):
            try:
                self.q_table = np.load(Q_TABLE_FILE)
                log.info("🧠 تم استرجاع الذاكرة السابقة. العقل مستعد للسوق الحقيقي.")
            except:
                self.q_table = np.zeros((3, 3))
        else:
            self.q_table = np.zeros((3, 3))
            log.info("🧠 تم تخليق عقل كمي جديد (بدأ من الصفر).")

    def save_memory(self):
        np.save(Q_TABLE_FILE, self.q_table)

    def _get_state(self, ofi):
        if ofi < -0.3: return 0  
        elif ofi > 0.3: return 2 
        return 1                 

    def act(self, ofi, current_price):
        state = self._get_state(ofi)
        
        if self.last_state is not None and self.last_action is not None:
            price_diff = current_price - self.last_price
            reward = 0
            if self.last_action == 0: reward = price_diff   
            elif self.last_action == 2: reward = -price_diff 
            
            best_future_q = np.max(self.q_table[state])
            self.q_table[self.last_state, self.last_action] += self.learning_rate * \
                (reward + self.discount_factor * best_future_q - self.q_table[self.last_state, self.last_action])

        if random.uniform(0, 1) < self.exploration_rate:
            action = random.randint(0, 2) 
        else:
            action = np.argmax(self.q_table[state])

        self.last_state = state
        self.last_action = action
        self.last_price = current_price
        
        return action

class ExecutionEngine:
    def __init__(self):
        self.position = None
        self.entry_price = 0.0
        self.size_usd = 0.0
        # توسيع الأهداف قليلاً لكي تناسب السوق الحقيقي
        self.take_profit_pct = 0.003
        self.stop_loss_pct = 0.002

    def evaluate(self, current_price, ai_action, recommended_size):
        if self.position:
            pnl_pct = (current_price - self.entry_price) / self.entry_price
            if self.position == 'SHORT': pnl_pct = -pnl_pct
                
            pnl_usd = self.size_usd * pnl_pct
            
            hit_tp = pnl_pct >= self.take_profit_pct
            hit_sl = pnl_pct <= -self.stop_loss_pct
            ai_reversal = (self.position == 'LONG' and ai_action == 2) or (self.position == 'SHORT' and ai_action == 0)

            if hit_tp or hit_sl or ai_reversal:
                reason = "🎯 هدف ربح" if hit_tp else ("🛑 وقف خسارة" if hit_sl else "🔄 انعكاس ذكي")
                log.info(f"⚙️ إغلاق {self.position} | السعر: {current_price:.4f} | السبب: {reason}")
                log.info(f"   💵 PnL: ${pnl_usd:+.2f}")
                log.info("=" * 60)
                
                # حفظ البيانات للداشبورد
                trade_data = pd.DataFrame([{
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "type": self.position,
                    "entry_price": self.entry_price,
                    "exit_price": current_price,
                    "size_usd": self.size_usd,
                    "pnl_usd": pnl_usd,
                    "pnl_pct": pnl_pct * 100,
                    "reason": reason.split()[1] 
                }])
                trade_data.to_csv(TRADES_FILE, mode='a', header=not os.path.exists(TRADES_FILE), index=False)
                
                # 🟢 السحر هنا: تحديث الذاكرة فوراً لكي تظهر الألوان في الداشبورد اللحظي
                if streamer and streamer.brain:
                    streamer.brain.save_memory()

                self.position = None
                return True
                
        elif recommended_size > 0:
            if ai_action == 0:
                self.position = 'LONG'
                self.entry_price = current_price
                self.size_usd = recommended_size
                log.info(f"🚀 دخول LONG 🟢 | السعر: {current_price:.4f}")
            elif ai_action == 2:
                self.position = 'SHORT'
                self.entry_price = current_price
                self.size_usd = recommended_size
                log.info(f"🚀 دخول SHORT 🔴 | السعر: {current_price:.4f}")
        return False

# 🟢 محرك الاتصال المباشر بالسوق الحقيقي
class LiveMarketStreamer:
    def __init__(self, symbol: str):
        self.symbol = symbol
        # الاتصال بمنصة Bybit لضمان الاستقرار في منطقتك
        self.exchange = ccxt.bybit({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
            }
        })
        self.order_flow = OrderFlowManager()
        self.risk_manager = RiskManager()
        self.brain = QuantBrain()
        self.execution = ExecutionEngine()
        self.trade_count = 0

    async def stream_ticks(self):
        log.info(f"⏳ جاري الاتصال المباشر بـ Bybit للزوج {self.symbol}...")
        await self.exchange.load_markets()
        log.info("✅ تم الاتصال بالسوق الحقيقي بنجاح! تدفق البيانات الحية بدأ...")
        
        while True:
            try:
                # استقبال سيل من الصفقات الحقيقية
                trades = await self.exchange.watch_trades(self.symbol)
                
                # معالجة كل صفقة في الدفعة المستلمة
                for trade in trades:
                    price = trade['price']
                    amount = trade['amount']
                    side = trade['side']
                    
                    if not side or not price or not amount: 
                        continue
                    
                    # التحليل والقرار في أجزاء من الثانية
                    ofi, cvd = self.order_flow.process_tick(price, amount, side)
                    action = self.brain.act(ofi, price)
                    
                    pos_size = 0
                    if action != 1 and not self.execution.position:
                        self.trade_count += 1
                        win_rate = min(0.5 + (0.005 * self.trade_count), 0.65) 
                        pos_size, _ = self.risk_manager.calculate_position_size(win_rate, 1.5)

                    self.execution.evaluate(price, action, pos_size)
                    
            except ccxt.NetworkError as e:
                log.warning(f"🌐 انقطاع لحظي في تدفق البيانات، جاري إعادة الاتصال... {e}")
                await asyncio.sleep(1)
            except Exception as e:
                log.error(f"❌ خطأ غير متوقع: {e}")
                break

    async def close(self):
        log.info("🛑 إغلاق الاتصال بالسوق الحقيقي وتنظيف الموارد...")
        await self.exchange.close()
        await asyncio.sleep(0.25)

streamer = None

async def main():
    global streamer
    streamer = LiveMarketStreamer("SOL/USDT")
    try:
        await streamer.stream_ticks()
    except KeyboardInterrupt:
        log.info("🛑 تم رصد أمر إيقاف يدهوي.")
    finally:
        if streamer:
            streamer.brain.save_memory()
            await streamer.close()

if __name__ == "__main__":
    asyncio.run(main())