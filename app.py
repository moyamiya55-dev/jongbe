import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytz
import streamlit as st
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass
from alpaca.trading.requests import GetAssetsRequest

st.set_page_config(page_title="미국종베", page_icon="🌙", layout="wide")

KEY = st.secrets["ALPACA_KEY"]
SECRET = st.secrets["ALPACA_SECRET"]
NY = pytz.timezone("America/New_York")
KST = pytz.timezone("Asia/Seoul")
COST = 0.015
client = StockHistoricalDataClient(KEY, SECRET)


@st.cache_data(ttl=86400)
def load_cases():
    return pd.read_csv("cases_app.csv")


@st.cache_data(ttl=86400)
def get_syms():
    assets = TradingClient(KEY, SECRET, paper=True).get_all_assets(
        GetAssetsRequest(asset_class=AssetClass.US_EQUITY))
    return [a.symbol for a in assets
            if a.tradable and a.exchange.value in ("NASDAQ", "NYSE", "AMEX")
            and "." not in a.symbol and "/" not in a.symbol]


def fetch(symbols, tf, start, end, bar=None, label=""):
    out, n = [], len(symbols)
    for i in range(0, n, 200):
        try:
            r = StockBarsRequest(symbol_or_symbols=symbols[i:i + 200], timeframe=tf,
                                 start=start, end=end, feed=DataFeed.SIP,
                                 adjustment=Adjustment.SPLIT)
            df = client.get_stock_bars(r).df
            if len(df):
                out.append(df.reset_index())
        except Exception:
            pass
        if bar is not None:
            bar.progress(min(1.0, (i + 200) / n), text=label)
    return pd.concat(out) if out else pd.DataFrame()


def find_candidates(drop_min, hi_max):
    syms = get_syms()
    now = datetime.now(NY)
    today = now.date()
    end = min(now - timedelta(minutes=16),
              NY.localize(datetime(today.year, today.month, today.day, 15, 59)))
    bar = st.progress(0.0, text="일봉 받는 중")
    dd = fetch(syms, TimeFrame.Day, now - timedelta(days=100), end, bar, "1/2 일봉 받는 중")
    dd["date"] = pd.to_datetime(dd["timestamp"], utc=True).dt.tz_convert("America/New_York").dt.date
    hist = dd[dd["date"] < today].sort_values("date")
    prev = hist.groupby("symbol")["close"].last().rename("전일종가")
    hi60 = hist.groupby("symbol").tail(60).groupby("symbol")["close"].max().rename("60일고점")
    intr = fetch(syms, TimeFrame(15, TimeFrameUnit.Minute),
                 NY.localize(datetime(today.year, today.month, today.day, 9, 30)), end,
                 bar, "2/2 오늘 분봉 받는 중")
    bar.empty()
    if intr.empty:
        return None, None
    intr = intr.sort_values("timestamp")
    cur = intr.groupby("symbol").agg(현재가=("close", "last"), 거래량=("volume", "sum"),
                                      기준시각=("timestamp", "last"))
    t = cur.join(prev).join(hi60).dropna()
    t["당일등락"] = t["현재가"] / t["전일종가"] - 1
    t["고점대비"] = t["현재가"] / t["60일고점"] - 1
    cand = t[(t["당일등락"] <= drop_min) & (t["고점대비"] <= hi_max) & (t["고점대비"] > -0.97)
             & (t["현재가"].between(0.1, 20)) & (t["거래량"] >= 100_000)].copy()
    cand["투매기준가"] = cand["현재가"] * 0.9
    asof = pd.Timestamp(cur["기준시각"].max()).tz_convert(KST).strftime("%m/%d %H:%M")
    return cand.sort_values("당일등락"), asof


def wilson(k, n, z=1.96):
    if n == 0:
        return 0, 0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0, c - h), min(1, c + h)


def similar(cs, drop, hiq):
    m = cs[(cs["drop1530"].sub(drop).abs() <= 0.08) & (cs["hi_q"].sub(hiq).abs() <= 0.15)]
    if len(m) < 20:
        m = cs[(cs["drop1530"].sub(drop).abs() <= 0.15) & (cs["hi_q"].sub(hiq).abs() <= 0.25)]
    return m


def pct(x):
    return f"{x * 100:.1f}%"


def color(p):
    return "#4ade80" if p >= 0.5 else ("#facc15" if p >= 0.3 else "#94a3b8")


def card(sym, r, cs, fx):
    m = similar(cs, r["당일등락"], r["고점대비"])
    n = len(m)
    if n == 0:
        return ""
    ks = {T: int((m["ah10"] >= T).sum()) for T in (0.05, 0.10, 0.20)}
    ps = {T: ks[T] / n for T in ks}
    dm = m[m["late_dump"] <= -0.10]
    if len(dm) < 8:
        g = "표본 부족"
    else:
        pnl = np.where(dm["ah10"] >= 0.05, 0.05, dm["exit10"] / dm["entry"] - 1) - COST
        hit = (dm["ah10"] >= 0.05).mean()
        if pnl.mean() >= 0.015 and hit >= 0.6:
            g = "A"
        elif pnl.mean() >= 0.008:
            g = "B"
        elif pnl.mean() > 0:
            g = "C"
        else:
            g = "패스"
    if len(dm):
        pnl = np.where(dm["ah10"] >= 0.05, 0.05, dm["exit10"] / dm["entry"] - 1) - COST
        dump_txt = (f"막판 -10% 투매 시: {len(dm)}건 · +5% 도달 {pct((dm['ah10'] >= 0.05).mean())}"
                    f" · 평균손익 {pnl.mean() * 100:+.1f}%")
    else:
        dump_txt = "막판 투매 유사 사례 없음"
    boxes = ""
    for T, lab in [(0.05, "C · +5% 이상"), (0.10, "B · +10% 이상"), (0.20, "A · +20% 이상")]:
        lo, hi = wilson(ks[T], n)
        boxes += (f"<div class='box'><div class='lab'>{lab}</div>"
                  f"<div class='val' style='color:{color(ps[T])}'>{pct(ps[T])}</div>"
                  f"<div class='ci'>95% 범위 {pct(lo)}~{pct(hi)}</div></div>")
    gcls = "g-none" if g in ("표본 부족", "패스") else "g-on"
    return (f"<div class='card'><div class='left'><div class='sym'>{sym}</div>"
            f"<div class='price'>${r['현재가']:.4f} <span class='krw'>({r['현재가'] * fx:,.0f}원)</span></div>"
            f"<div class='meta'>당일 {pct(r['당일등락'])} · 고점대비 {pct(r['고점대비'])}</div>"
            f"<div class='meta'>투매기준가 ${r['투매기준가']:.4f}</div>"
            f"<span class='grade {gcls}'>{g}</span>"
            f"<div class='meta'>유사사례 {n}건 · 중앙값 {m['ah10'].median() * 100:+.1f}%</div></div>"
            f"<div class='right'><div class='boxes'>{boxes}</div>"
            f"<div class='dump'>{dump_txt}</div></div></div>")


CSS = """<style>
.wrap{display:flex;flex-direction:column;gap:10px}
.card{display:flex;gap:16px;background:#0f1b33;color:#e2e8f0;border:1px solid #1e3a5f;border-radius:12px;padding:14px;flex-wrap:wrap}
.left{min-width:180px;flex:1}.right{flex:2;min-width:260px}
.sym{font-size:20px;font-weight:700}.price{font-size:16px;margin:4px 0}.krw{color:#94a3b8;font-size:13px}
.meta{color:#94a3b8;font-size:12px;margin:2px 0}
.grade{display:inline-block;margin:6px 0;padding:2px 10px;border-radius:6px;font-size:12px;border:1px solid}
.g-none{color:#94a3b8;border-color:#475569}.g-on{color:#4ade80;border-color:#4ade80}
.boxes{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
.box{background:#13254a;border:1px solid #1e3a5f;border-radius:8px;padding:8px}
.lab{font-size:11px;color:#94a3b8}.val{font-size:18px;font-weight:700}.ci{font-size:10px;color:#64748b}
.dump{margin-top:8px;font-size:12px;color:#fbbf24}
</style>"""

st.title("🌙 미국 종베")
st.caption("매일 04:15 KST 조회 → 04:30 앱 가격 메모 → 04:55 막판 -10% 투매 확인 → 종베 스킬 → 04:59 매수 · 05:00~05:10 매도")

with st.sidebar:
    fx = st.number_input("원/달러 환율", value=1350, step=10)
    drop_min = st.slider("당일 하락 기준", -0.50, -0.05, -0.10, 0.01, format="%.2f")
    hi_max = st.slider("60일 고점 대비 기준", -0.90, -0.20, -0.40, 0.05, format="%.2f")
    top_n = st.slider("카드 개수", 5, 20, 10)

if st.button("🔍 후보 조회 (2~4분)", type="primary", use_container_width=True):
    with st.spinner("데이터 받는 중..."):
        cand, asof = find_candidates(drop_min, hi_max)
    st.session_state["cand"], st.session_state["asof"] = cand, asof

cand = st.session_state.get("cand")
if "cand" in st.session_state:
    if cand is None:
        st.warning("오늘 장 데이터가 없어요 (휴장일이거나 개장 전).")
    else:
        st.subheader(f"후보 {len(cand)}개 · 데이터 기준 {st.session_state['asof']} KST (15분 지연)")
        cs = load_cases()
        html = "".join(card(sym, r, cs, fx) for sym, r in cand.head(top_n).iterrows())
        st.markdown(CSS + "<div class='wrap'>" + html + "</div>", unsafe_allow_html=True)
        with st.expander("전체 후보 표"):
            st.dataframe(cand[["현재가", "당일등락", "고점대비", "거래량", "투매기준가"]].round(4))

with st.expander("등급 기준"):
    st.markdown("- **A**: 막판 투매 시 평균손익 +1.5% 이상, +5% 도달 60% 이상\n"
                "- **B**: 평균손익 +0.8% 이상\n- **C**: 평균손익 플러스\n"
                "- **패스**: 투매가 나와도 마이너스\n- **표본 부족**: 비슷한 투매 사례 8건 미만\n\n"
                "과거 데이터 기준 참고용이에요. 최종 판단은 호가창과 공시 확인 후에 하세요.")
