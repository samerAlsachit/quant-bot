import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px

# إعداد الصفحة
st.set_page_config(page_title="Samer Quant AI", page_icon="🧠", layout="wide")

Q_TABLE_FILE = "memory/q_table.npy"
TRADES_FILE = "memory/trades_history.csv"

# دالة قراءة بيانات مضادة للأخطاء (Bulletproof)
def load_data():
    q_table = np.zeros((3, 3))
    trades = pd.DataFrame()
    
    # تحميل العقل
    if os.path.exists(Q_TABLE_FILE):
        try:
            q_table = np.load(Q_TABLE_FILE)
        except Exception:
            pass # تجاهل الخطأ والبدء بمصفوفة أصفار
            
    # تحميل الصفقات
    if os.path.exists(TRADES_FILE):
        try:
            # تجنب انهيار النظام إذا كان الملف فارغاً تماماً
            if os.path.getsize(TRADES_FILE) > 0: 
                trades = pd.read_csv(TRADES_FILE)
                if not trades.empty and 'timestamp' in trades.columns:
                    trades['timestamp'] = pd.to_datetime(trades['timestamp'])
        except Exception as e:
            st.error(f"⚠️ خطأ في قراءة ملف الصفقات: {e}")
            
    return q_table, trades

st.sidebar.title("🧠 Samer Quant AI")
st.sidebar.caption("v5.1 — Bulletproof Edition")
auto_refresh = st.sidebar.checkbox("🔄 تحديث تلقائي (كل 5 ثوان)", value=True)

# تحميل البيانات بأمان
q_table, trades = load_data()

st.title("لوحة تحكم الذكاء الاصطناعي الكمي (Quant Dashboard)")

# 1. إحصائيات الأداء
if not trades.empty and 'pnl_usd' in trades.columns:
    total_trades = len(trades)
    win_trades = len(trades[trades['pnl_usd'] > 0])
    win_rate = (win_trades / total_trades) * 100 if total_trades > 0 else 0
    total_pnl = trades['pnl_usd'].sum()
    best_trade = trades['pnl_usd'].max()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("إجمالي الصفقات", total_trades)
    col2.metric("معدل النجاح (Win Rate)", f"{win_rate:.1f}%")
    col3.metric("صافي الأرباح/الخسائر", f"${total_pnl:.2f}")
    col4.metric("أفضل صفقة", f"${best_trade:.2f}")
    st.divider()

# 2. عرض مصفوفة التعلم (Q-Table)
st.subheader("🧠 دماغ البوت: خريطة الحرارة للتعلم المعزز")
fig_q = px.imshow(
    q_table,
    labels=dict(x="القرار (Action)", y="حالة السوق (OFI State)", color="الوزن (Weight)"),
    x=['شراء (Long)', 'انتظار (Hold)', 'بيع (Short)'],
    y=['ضغط بيعي (-1)', 'متوازن (0)', 'ضغط شرائي (+1)'],
    color_continuous_scale="Viridis",
    aspect="auto",
    text_auto=".4f"
)
st.plotly_chart(fig_q, use_container_width=True)

# 3. تحليل الصفقات
if not trades.empty and 'pnl_usd' in trades.columns:
    st.subheader("📈 التوزيع الإحصائي التراكمي")
    col_a, col_b = st.columns(2)
    
    with col_a:
        trades['cumulative_pnl'] = trades['pnl_usd'].cumsum()
        fig_growth = px.line(trades, x='timestamp', y='cumulative_pnl', title="نمو المحفظة اللحظي")
        st.plotly_chart(fig_growth, use_container_width=True)
        
    with col_b:
        if 'reason' in trades.columns:
            fig_reasons = px.pie(trades, names='reason', title="أسباب إغلاق الصفقات")
            st.plotly_chart(fig_reasons, use_container_width=True)
        
    st.subheader("📜 سجل العمليات (Order Log)")
    st.dataframe(trades.sort_values(by="timestamp", ascending=False).head(20), use_container_width=True)
else:
    st.info("💡 النظام يعمل بنجاح. بانتظار إغلاق الصفقات لرسم الإحصائيات...")

# التحديث التلقائي
if auto_refresh:
    import time
    time.sleep(5)
    st.rerun()