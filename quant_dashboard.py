import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px

st.set_page_config(page_title="Samer Quant v7.0", page_icon="🧠", layout="wide")

Q_TABLE_FILE = "memory/q_table.npy"
TRADES_FILE = "memory/trades_history.csv"

def load_data():
    # تحميل العقل
    q_table = None
    if os.path.exists(Q_TABLE_FILE):
        try:
            raw_q = np.load(Q_TABLE_FILE)
            if len(raw_q.shape) == 4:
                # ضغط المصفوفة الرباعية إلى ثنائية الأبعاد (OFI vs Action) لسهولة العرض
                q_table = np.mean(raw_q, axis=(1, 2))
        except: pass
    
    if q_table is None: q_table = np.zeros((3, 3))
            
    # تحميل الصفقات
    trades = pd.DataFrame()
    if os.path.exists(TRADES_FILE) and os.path.getsize(TRADES_FILE) > 0:
        try:
            trades = pd.read_csv(TRADES_FILE)
            if not trades.empty and 'timestamp' in trades.columns:
                trades['timestamp'] = pd.to_datetime(trades['timestamp'])
        except: pass
            
    return q_table, trades

st.sidebar.title("🧠 Quant AI v7.0")
st.sidebar.caption("4D Tensor & Bayesian Smoothing")
auto_refresh = st.sidebar.checkbox("🔄 تحديث تلقائي", value=True)

q_table, trades = load_data()

st.title("لوحة التحكم الكمية المتقدمة")

if not trades.empty and 'pnl_usd' in trades.columns:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("إجمالي الصفقات", len(trades))
    win_rate = (len(trades[trades['pnl_usd'] > 0]) / len(trades)) * 100
    col2.metric("معدل النجاح (Win Rate)", f"{win_rate:.1f}%")
    col3.metric("صافي الأرباح/الخسائر", f"${trades['pnl_usd'].sum():.2f}")
    col4.metric("أفضل صفقة", f"${trades['pnl_usd'].max():.2f}")
    st.divider()

st.subheader("🧠 دماغ البوت: الوعي السعري المجمع (Aggregated Q-Table)")
st.markdown("يتم عرض متوسط أوزان اتخاذ القرار بناءً على حالة تدفق الأوامر (OFI) متجاهلين أبعاد التقلب والزخم لتسهيل القراءة.")

fig_q = px.imshow(
    q_table,
    labels=dict(x="القرار (Action)", y="حالة (OFI)", color="الوزن"),
    x=['شراء (Long)', 'انتظار (Hold)', 'بيع (Short)'],
    y=['ضغط بيعي (-1)', 'متوازن (0)', 'ضغط شرائي (+1)'],
    color_continuous_scale="Viridis",
    aspect="auto",
    text_auto=".4f"
)
st.plotly_chart(fig_q, use_container_width=True)

if not trades.empty and 'pnl_usd' in trades.columns:
    col_a, col_b = st.columns(2)
    with col_a:
        trades['cumulative_pnl'] = trades['pnl_usd'].cumsum()
        fig_growth = px.line(trades, x='timestamp', y='cumulative_pnl', title="نمو المحفظة")
        st.plotly_chart(fig_growth, use_container_width=True)
    with col_b:
        if 'reason' in trades.columns:
            fig_reasons = px.pie(trades, names='reason', title="محفزات الإغلاق")
            st.plotly_chart(fig_reasons, use_container_width=True)
            
    st.subheader("📜 سجل العمليات")
    st.dataframe(trades.sort_values(by="timestamp", ascending=False).head(20), use_container_width=True)

if auto_refresh:
    import time
    time.sleep(5)
    st.rerun()