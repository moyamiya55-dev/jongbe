import math
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytz
import requests
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
KIS_KEY = st.secrets.get("KIS_APP_KEY", "")
KIS_SECRET = st.secrets.get("KIS_APP_SECRET", "")
KIS_BASE = "https://openapi.koreainvestment.com:9443"
EXMAP = {"NASDAQ": "NAS", "NYSE": "NYS", "AMEX": "AMS"}

NY = pytz.timezone("America/New_York")
KST = pytz.timezone("Asia/Seoul")
COST = 0.015
DUMP = -0.10
client = StockHistoricalDataClient(KEY, SECRET)


# ---------------- 데이터 ----------------
@st.cache_resource
def store():
    """세션이 끊겨도 같은 날 조회 결과를 다시 쓰기 위한 서버 저장소"""
    return {}


@st.cache_data(ttl=86400)
def load_cases():
    return pd.read_csv("cases_app.csv")


@st.cache_data(ttl=86400)
def get_assets():
    assets = TradingClient(KEY, SECRET, paper=True).get_all_assets(
        GetAssetsRequest(asset_class=AssetClass.US_EQUITY))
    return {a.symbol: a.exchange.value for a in assets
            if a.tradable and a.exchange.value in EXMAP
            and "." not in a.symbol and "/" not in a.symbol}


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
            bar.progress(min(1.0, (i + 200) / max(n, 1)), text=label)
    return pd.concat(out) if out else pd.DataFrame()


def find_candidates(drop_min, hi_max):
    syms = list(get_assets().keys())
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


def price_at_1530(symbols):
    """알파카 1분봉에서 15:30 ET(04:30 KST) 가격"""
    now = datetime.now(NY)
    d = now.date()
    start = NY.localize(datetime(d.year, d.month, d.day, 15, 25))
    end = min(now - timedelta(minutes=16), NY.localize(datetime(d.year, d.month, d.day, 15, 31)))
    if end <= start:
        return {}
    mb = fetch(symbols, TimeFrame.Minute, start, end)
    if mb.empty:
        return {}
    mb["hm"] = pd.to_datetime(mb["timestamp"], utc=True).dt.tz_convert("America/New_York").dt.strftime("%H:%M")
    mb = mb[mb["hm"] <= "15:30"].sort_values("timestamp")
    return mb.groupby("symbol")["close"].last().to_dict()


# ---------------- 한투 ----------------
@st.cache_data(ttl=60 * 60 * 20, show_spinner=False)
def kis_token():
    r = requests.post(f"{KIS_BASE}/oauth2/tokenP", timeout=10,
                      json={"grant_type": "client_credentials",
                            "appkey": KIS_KEY, "appsecret": KIS_SECRET})
    j = r.json()
    if "access_token" not in j:
        raise RuntimeError(j.get("error_description") or j.get("msg1") or str(j)[:120])
    return j["access_token"]


def kis_price(sym, excd):
    h = {"content-type": "application/json; charset=utf-8",
         "authorization": f"Bearer {kis_token()}",
         "appkey": KIS_KEY, "appsecret": KIS_SECRET,
         "tr_id": "HHDFS00000300", "custtype": "P"}
    try:
        r = requests.get(f"{KIS_BASE}/uapi/overseas-price/v1/quotations/price",
                         headers=h, params={"AUTH": "", "EXCD": excd, "SYMB": sym}, timeout=10)
        j = r.json()
    except Exception as e:
        return None, str(e)[:60]
    if j.get("rt_cd") != "0":
        return None, j.get("msg1", "조회 실패")
    try:
        return float(j["output"]["last"]), None
    except (KeyError, TypeError, ValueError):
        return None, "가격 없음"


# ---------------- 카드 ----------------
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
        return "", "사례 없음", ""
    ks = {T: int((m["ah10"] >= T).sum()) for T in (0.05, 0.10, 0.20)}
    ps = {T: ks[T] / n for T in ks}
    dm = m[m["late_dump"] <= DUMP]
    if len(dm):
        pnl = np.where(dm["ah10"] >= 0.05, 0.05, dm["exit10"] / dm["entry"] - 1) - COST
        hit = (dm["ah10"] >= 0.05).mean()
        dump_txt = (f"막판 -10% 투매 시: {len(dm)}건 · +5% 도달 {pct(hit)}"
                    f" · 평균손익 {pnl.mean() * 100:+.1f}%")
    else:
        pnl, hit = np.array([]), 0
        dump_txt = "막판 투매 유사 사례 없음"
    if len(dm) < 8:
        g = "표본 부족"
    elif pnl.mean() >= 0.015 and hit >= 0.6:
        g = "A"
    elif pnl.mean() >= 0.008:
        g = "B"
    elif pnl.mean() > 0:
        g = "C"
    else:
        g = "패스"
    boxes = ""
    for T, lab in [(0.05, "C · +5% 이상"), (0.10, "B · +10% 이상"), (0.20, "A · +20% 이상")]:
        lo, hi = wilson(ks[T], n)
        boxes += (f"<div class='box'><div class='lab'>{lab}</div>"
                  f"<div class='val' style='color:{color(ps[T])}'>{pct(ps[T])}</div>"
                  f"<div class='ci'>95% 범위 {pct(lo)}~{pct(hi)}</div></div>")
    gcls = "g-none" if g in ("표본 부족", "패스") else "g-on"
    html = (f"<div class='card'><div class='left'><div class='sym'>{sym}</div>"
            f"<div class='price'>${r['현재가']:.4f} <span class='krw'>({r['현재가'] * fx:,.0f}원)</span></div>"
            f"<div class='meta'>당일 {pct(r['당일등락'])} · 고점대비 {pct(r['고점대비'])}</div>"
            f"<div class='meta'>투매기준가 ${r['투매기준가']:.4f}</div>"
            f"<span class='grade {gcls}'>{g}</span>"
            f"<div class='meta'>유사사례 {n}건 · 중앙값 {m['ah10'].median() * 100:+.1f}%</div></div>"
            f"<div class='right'><div class='boxes'>{boxes}</div>"
            f"<div class='dump'>{dump_txt}</div></div></div>")
    return html, g, dump_txt


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
.hit{background:#14361f;border:1px solid #22c55e;border-radius:12px;padding:12px;margin:6px 0;color:#e2e8f0}
.hit b{font-size:18px}
</style>"""

# ---------------- 화면 ----------------
st.title("🌙 미국 종베")
st.caption("04:15 후보 조회 → 04:50~04:58 투매 체크 → 종베 스킬로 공시 확인 → 04:59 매수 · 05:00~05:10 매도")

with st.sidebar:
    fx = st.number_input("원/달러 환율", value=1350, step=10)
    drop_min = st.slider("당일 하락 기준", -0.50, -0.05, -0.10, 0.01, format="%.2f")
    hi_max = st.slider("60일 고점 대비 기준", -0.90, -0.20, -0.40, 0.05, format="%.2f")
    top_n = st.slider("카드 개수", 5, 20, 10)
    st.divider()
    if st.button("한투 연결 테스트"):
        if not KIS_KEY:
            st.error("Secrets에 KIS_APP_KEY / KIS_APP_SECRET이 없어요.")
        else:
            try:
                p, err = kis_price("AAPL", "NAS")
                if p:
                    st.success(f"연결 OK · AAPL ${p}")
                else:
                    st.error(f"조회 실패: {err}")
            except Exception as e:
                st.error(f"토큰 발급 실패: {e}")

today_key = datetime.now(NY).strftime("%Y-%m-%d")

if st.button("🔍 후보 조회 (2~4분)", type="primary", use_container_width=True):
    with st.spinner("데이터 받는 중..."):
        cand, asof = find_candidates(drop_min, hi_max)
    store()[today_key] = (cand, asof)
    st.session_state["cand"], st.session_state["asof"] = cand, asof
elif "cand" not in st.session_state and today_key in store():
    st.session_state["cand"], st.session_state["asof"] = store()[today_key]

cand = st.session_state.get("cand")
info = {}
if "cand" in st.session_state:
    if cand is None:
        st.warning("오늘 장 데이터가 없어요 (휴장일이거나 개장 전).")
    else:
        st.subheader(f"후보 {len(cand)}개 · 데이터 기준 {st.session_state['asof']} KST (15분 지연)")
        cs = load_cases()
        html = ""
        for sym, r in cand.head(top_n).iterrows():
            h, g, dtxt = card(sym, r, cs, fx)
            html += h
            info[sym] = (g, dtxt)

        # ---------- 투매 체크 ----------
        st.divider()
        st.subheader("⏱ 투매 체크")
        st.caption("04:47 이후에 누르세요. 04:30 가격(알파카)과 지금 실시간 가격(한투)을 비교해서 -10% 이상 빠진 종목을 찾아요.")
        if st.button("⏱ 투매 체크 실행", use_container_width=True):
            if not KIS_KEY:
                st.error("Secrets에 KIS_APP_KEY / KIS_APP_SECRET을 넣어주세요.")
            else:
                syms = list(cand.head(top_n).index)
                with st.spinner("04:30 가격 확인 중..."):
                    p30 = price_at_1530(syms)
                if not p30:
                    st.warning("04:30 가격이 아직 없어요. 04:47 이후에 다시 눌러주세요.")
                else:
                    ex = get_assets()
                    rows = []
                    with st.spinner("실시간 가격 조회 중..."):
                        for s in syms:
                            if s not in p30:
                                continue
                            now_p, err = kis_price(s, EXMAP.get(ex.get(s, ""), "NAS"))
                            time.sleep(0.08)
                            chg = now_p / p30[s] - 1 if now_p else None
                            g, dtxt = info.get(s, ("", ""))
                            rows.append({"종목": s, "등급": g, "04:30": p30[s],
                                         "지금(실시간)": now_p, "04:30 대비": chg,
                                         "투매": "✅" if chg is not None and chg <= DUMP else "",
                                         "메모": err or dtxt})
                    res = pd.DataFrame(rows).sort_values("04:30 대비", na_position="last")
                    st.session_state["check"] = res
                    st.session_state["check_at"] = datetime.now(KST).strftime("%H:%M:%S")

        res = st.session_state.get("check")
        if res is not None and len(res):
            st.caption(f"마지막 체크 {st.session_state['check_at']} KST")
            hits = res[res["투매"] == "✅"]
            if len(hits):
                for _, h in hits.iterrows():
                    st.markdown(
                        f"<div class='hit'><b>{h['종목']}</b> · 등급 {h['등급']} · "
                        f"04:30 ${h['04:30']:.4f} → 지금 ${h['지금(실시간)']:.4f} "
                        f"(<b>{h['04:30 대비'] * 100:+.1f}%</b>)<br>"
                        f"<span style='color:#fbbf24;font-size:12px'>{h['메모']}</span><br>"
                        f"<span style='font-size:12px'>👉 종베 스킬로 장후 공모 위험 확인 후 진입</span></div>",
                        unsafe_allow_html=True)
            else:
                st.info("아직 -10% 투매가 나온 종목이 없어요. 04:55쯤 한 번 더 눌러보세요.")
            show = res.copy()
            show["04:30 대비"] = show["04:30 대비"].map(lambda x: f"{x * 100:+.1f}%" if pd.notna(x) else "-")
            st.dataframe(show, hide_index=True, use_container_width=True)

        st.divider()
        st.markdown(CSS + "<div class='wrap'>" + html + "</div>", unsafe_allow_html=True)
        with st.expander("전체 후보 표"):
            st.dataframe(cand[["현재가", "당일등락", "고점대비", "거래량", "투매기준가"]].round(4))

st.markdown(CSS, unsafe_allow_html=True)
with st.expander("등급 기준"):
    st.markdown("- **A**: 막판 투매 시 평균손익 +1.5% 이상, +5% 도달 60% 이상\n"
                "- **B**: 평균손익 +0.8% 이상\n- **C**: 평균손익 플러스\n"
                "- **패스**: 투매가 나와도 마이너스\n- **표본 부족**: 비슷한 투매 사례 8건 미만\n\n"
                "등급은 막판 -10% 투매가 실제로 나왔을 때의 과거 성적이에요. "
                "과거 데이터 기준 참고용이고, 최종 판단은 호가창과 공시 확인 후에 하세요.")
