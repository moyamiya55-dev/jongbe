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
    # 알파카 라이브러리 형식 검사를 거치지 않고 원본 목록을 직접 받음
    r = requests.get("https://paper-api.alpaca.markets/v2/assets", timeout=30,
                     headers={"APCA-API-KEY-ID": KEY, "APCA-API-SECRET-KEY": SECRET},
                     params={"asset_class": "us_equity", "status": "active"})
    r.raise_for_status()
    out = {}
    for a in r.json():
        sym, ex = str(a.get("symbol", "")), str(a.get("exchange", ""))
        if a.get("tradable") and ex in EXMAP and sym and "." not in sym and "/" not in sym:
            out[sym] = ex
    return out


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

# ---------------- 국장식 종베 (신정재 규칙) ----------------
KR = dict(vol_x=3.0, min_dollar=5_000_000, near_hi=-0.10, wick=0.25,
          adj_min=5, adj_max=30, min_price=5.0, max_price=20.0, max_vol_x=30.0, max_cards=15, min_up=0.05)
N_RULES = 6


def kst_of(hh, mm):
    d = datetime.now(NY).date()
    return NY.localize(datetime(d.year, d.month, d.day, hh, mm)).astimezone(KST).strftime("%H:%M")


def kr_scan():
    """일봉 + 오늘 15분봉으로 6개 규칙 판정"""
    syms = list(get_assets().keys())
    now = datetime.now(NY)
    today = now.date()
    end = min(now - timedelta(minutes=16),
              NY.localize(datetime(today.year, today.month, today.day, 15, 59)))
    bar = st.progress(0.0, text="일봉 받는 중")
    dd = fetch(syms, TimeFrame.Day, now - timedelta(days=130), end, bar, "1/2 일봉 받는 중")
    intr = fetch(syms, TimeFrame(15, TimeFrameUnit.Minute),
                 NY.localize(datetime(today.year, today.month, today.day, 9, 30)), end,
                 bar, "2/2 오늘 분봉 받는 중")
    bar.empty()
    if dd.empty or intr.empty:
        return None, None
    intr = intr.sort_values("timestamp")
    g = intr.groupby("symbol")
    tb = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(),
                       "low": g["low"].min(), "close": g["close"].last(),
                       "volume": g["volume"].sum()})
    tb = tb[(tb["close"] >= KR["min_price"]) & (tb["close"] <= KR["max_price"]) & (tb["close"] * tb["volume"] >= KR["min_dollar"])]
    asof = pd.Timestamp(intr["timestamp"].max()).tz_convert(KST).strftime("%m/%d %H:%M")

    dd["date"] = pd.to_datetime(dd["timestamp"], utc=True).dt.tz_convert("America/New_York").dt.date
    hist = dd[(dd["date"] < today) & dd["symbol"].isin(tb.index)].sort_values(["symbol", "date"])
    rows = []
    for sym, h in hist.groupby("symbol"):
        if len(h) < 25:
            continue
        t = tb.loc[sym]
        h60 = h.tail(60).reset_index(drop=True)
        pk = h60["high"].idxmax()
        hi = h60.loc[pk, "high"]
        adj_days = len(h60) - pk          # 고점 다음날부터 오늘까지 거래일 수
        avg20 = h["volume"].tail(20).mean()
        vol_x = t["volume"] / avg20 if avg20 > 0 else 0
        gap_hi = t["close"] / hi - 1
        rng = t["high"] - t["low"]
        wick = (t["high"] - t["close"]) / rng if rng > 0 else 0
        green = t["close"] > t["open"]
        ma5 = (h["close"].tail(4).sum() + t["close"]) / 5
        lows = h["low"].tail(10).values
        higher_low = len(lows) == 10 and lows[5:].min() > lows[:5].min()
        ok = {
            "거래량": KR["vol_x"] <= vol_x < KR["max_vol_x"],
            "전고점근접": gap_hi >= KR["near_hi"],
            "깔끔양봉": bool(green and wick <= KR["wick"]),
            "기간조정": KR["adj_min"] <= adj_days <= KR["adj_max"],
            "5일선": t["close"] >= ma5,
            "상승률": t["close"] / h["close"].iloc[-1] - 1 >= KR["min_up"],
        }
        rows.append({"종목": sym, "현재가": t["close"],
                     "당일등락": t["close"] / h["close"].iloc[-1] - 1,
                     "거래량배수": vol_x, "고점대비": gap_hi, "위꼬리": wick,
                     "조정일수": adj_days, "저점높임": higher_low,
                     "저가": t["low"], "통과수": sum(ok.values()), "탈락": ", ".join(k for k, v in ok.items() if not v)})
    if not rows:
        return pd.DataFrame(), asof
    res = pd.DataFrame(rows).set_index("종목")
    return res.sort_values(["통과수", "거래량배수"], ascending=False), asof


def kr_minute_check(symbols):
    """1분봉: 마감 전 30분 동안 당일 저점 안 깸 + 20선 위 + 20선 상승"""
    now = datetime.now(NY)
    d = now.date()
    start = NY.localize(datetime(d.year, d.month, d.day, 9, 30))
    end = min(now - timedelta(minutes=16), NY.localize(datetime(d.year, d.month, d.day, 15, 59)))
    symbols = list(symbols)
    if not symbols or end <= start:
        return {}, ""
    mb = fetch(list(symbols), TimeFrame.Minute, start, end)
    if mb.empty:
        return {}, ""
    mb = mb.sort_values("timestamp")
    asof = pd.Timestamp(mb["timestamp"].max()).tz_convert(KST).strftime("%H:%M")
    out = {}
    for sym, m in mb.groupby("symbol"):
        if len(m) < 40:
            out[sym] = "❌ 분봉 부족"
            continue
        c = m["close"].reset_index(drop=True)
        ma = c.rolling(20).mean()
        low_kept = m["low"].tail(30).min() > m["low"].iloc[:-30].min()
        above = c.iloc[-1] >= ma.iloc[-1]
        rising = ma.iloc[-1] > ma.iloc[-11]
        bad = [n for n, v in [("최근 30분 당일저점 갱신", low_kept), ("20선 아래", above),
                              ("20선 하락", rising)] if not v]
        out[sym] = "✅ 저점 유지 · 20선 위 우상향" if not bad else "❌ " + " · ".join(bad)
    return out, asof


def kr_history(n_days):
    """최근 n_days 거래일 동안 6개 규칙에 걸렸을 종목 + 다음날 결과 (일봉 기준 재계산)"""
    syms = list(get_assets().keys())
    now = datetime.now(NY)
    bar = st.progress(0.0, text="과거 일봉 받는 중")
    dd = fetch(syms, TimeFrame.Day, now - timedelta(days=int(n_days * 1.5) + 110),
               now - timedelta(minutes=16), bar, "과거 일봉 받는 중")
    bar.empty()
    if dd.empty:
        return pd.DataFrame()
    dd["date"] = pd.to_datetime(dd["timestamp"], utc=True).dt.tz_convert("America/New_York").dt.date
    dd = dd[dd["date"] < now.date()].sort_values(["symbol", "date"]).reset_index(drop=True)
    g = dd.groupby("symbol")
    dd["n"] = g.cumcount()
    dd["pv"] = g["volume"].shift(1)
    dd["pc"] = g["close"].shift(1)
    dd["avg20"] = dd.groupby("symbol")["pv"].rolling(20).mean().reset_index(level=0, drop=True)
    dd["ma5"] = g["close"].rolling(5).mean().reset_index(level=0, drop=True)
    for c in ("open", "high", "low", "close"):
        dd["다음" + c] = g[c].shift(-1)
    days = sorted(dd["date"].unique())[-n_days:]
    rng = (dd["high"] - dd["low"]).replace(0, np.nan)
    m = (dd["date"].isin(days) & (dd["n"] >= 25) & (dd["close"] >= KR["min_price"]) & (dd["close"] <= KR["max_price"])
         & (dd["close"] * dd["volume"] >= KR["min_dollar"])
         & (dd["volume"] >= KR["vol_x"] * dd["avg20"]) & (dd["volume"] < KR["max_vol_x"] * dd["avg20"]) & (dd["close"] > dd["open"])
         & ((dd["high"] - dd["close"]) / rng <= KR["wick"]) & (dd["close"] >= dd["ma5"]) & (dd["close"] >= dd["pc"] * (1 + KR["min_up"])))
    rows = []
    for i in dd.index[m]:
        n, sym = int(dd.at[i, "n"]), dd.at[i, "symbol"]
        h60 = dd.loc[i - min(n, 60):i - 1]
        pk = int(np.argmax(h60["high"].values))
        hi = h60["high"].iloc[pk]
        adj = len(h60) - pk
        c = dd.at[i, "close"]
        if c / hi - 1 < KR["near_hi"] or not (KR["adj_min"] <= adj <= KR["adj_max"]):
            continue
        rows.append({"날짜": dd.at[i, "date"], "종목": sym, "매수가": c,
                     "거래량배수": dd.at[i, "volume"] / dd.at[i, "avg20"],
                     "다음날시가": dd.at[i, "다음open"] / c - 1,
                     "다음날고가": dd.at[i, "다음high"] / c - 1,
                     "다음날종가": dd.at[i, "다음close"] / c - 1,
                     "다음날저가": dd.at[i, "다음low"] / c - 1,
                     "손절선": dd.at[i, "low"] / c - 1,
                     "전일종가": dd.at[i, "pc"]})
    return pd.DataFrame(rows)


def kr_exit_table(d, tgt=0.05):
    o, h, lo, c, sl = d["다음날시가"], d["다음날고가"], d["다음날저가"], d["다음날종가"], d["손절선"]
    hit = h >= tgt
    a = o
    b = np.where(o >= tgt, o, np.where(hit, tgt, c))
    stop_hit = lo <= sl
    cc = np.where(o >= tgt, o, np.where(o <= sl, o, np.where(stop_hit, sl, np.where(hit, tgt, c))))
    rows = []
    for name, r in [("① 시가 매도", a), ("② +5% 지정가, 못 닿으면 종가", b),
                    ("③ +5% 지정가 + 매수일 저가 이탈 손절", cc)]:
        r = pd.Series(r, dtype=float)
        rows.append({"방식": name, "평균": f"{r.mean() * 100:+.2f}%", "중앙값": f"{r.median() * 100:+.1f}%",
                     "이긴 비율": pct((r > 0).mean()), "최악": f"{r.min() * 100:+.1f}%"})
    return pd.DataFrame(rows)


def kr_bucket_tables(d, tgt=0.05):
    d = d.copy()
    d["당일상승"] = d["매수가"] / d["전일종가"] - 1
    hit = d["다음날고가"] >= tgt
    d["결과"] = np.where(d["다음날시가"] >= tgt, d["다음날시가"], np.where(hit, tgt, d["다음날종가"]))
    d["도달"] = hit
    specs = {
        "가격대": ("매수가", [0, 5, 20, 50, 1e9], ["$1~5", "$5~20", "$20~50", "$50+"]),
        "거래량 배수": ("거래량배수", [0, 5, 10, 30, 1e9], ["3~5배", "5~10배", "10~30배", "30배+"]),
        "당일 상승폭": ("당일상승", [0, 0.10, 0.20, 0.40, 1e9], ["5~10%", "10~20%", "20~40%", "40%+"]),
    }
    out = {}
    for name, (col, bins, labels) in specs.items():
        d["구간"] = pd.cut(d[col], bins=bins, labels=labels, right=False)
        rows = []
        for lab in labels:
            g = d[d["구간"] == lab]
            if not len(g):
                continue
            miss = g.loc[~g["도달"], "결과"]
            rows.append({"구간": lab, "건수": len(g), "+5% 도달률": pct(g["도달"].mean()),
                         "못 닿을 때 평균": f"{miss.mean() * 100:+.1f}%" if len(miss) else "-",
                         "전체 평균": f"{g['결과'].mean() * 100:+.2f}%"})
        out[name] = pd.DataFrame(rows)
    return out


# ---------------- 화면 ----------------
st.title("🌙 미국 종베")
tab1, tab2 = st.tabs(["🌙 투매 종베", "📈 국장식 종베"])

with tab1:
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

with tab2:
    st.caption(f"오늘 기준(한국시간) 관찰 {kst_of(15, 40)}~ · 매수 {kst_of(15, 55)}~{kst_of(15, 58)} · "
               "데이터가 15분 늦어서 조회는 관찰 시작 전에, 1분봉 재확인은 매수 직전에")
    cc1, cc2 = st.columns([1, 1])
    low_px = cc1.toggle("$1~5 포함", value=True, key="kr_lowpx")
    up_pct = cc2.radio("당일 상승 하한", [5, 10, 15, 20], horizontal=True, key="kr_up",
                       format_func=lambda x: f"+{x}%")
    KR["min_up"] = up_pct / 100
    if low_px:
        KR["min_price"] = 1.0
    kr_key = f"kr_{today_key}_{KR['min_price']}_{up_pct}"

    if st.button("🔍 국장식 후보 조회 (2~4분)", type="primary", use_container_width=True, key="kr_btn"):
        with st.spinner("데이터 받는 중..."):
            kres, kasof = kr_scan()
        mins, masof = ({}, "")
        if kres is not None and len(kres):
            mins, masof = kr_minute_check(kres[kres["통과수"] == N_RULES].head(KR["max_cards"]).index)
        store()[kr_key] = (kres, kasof, mins, masof)
    if kr_key in store():
        kres, kasof, mins, masof = store()[kr_key]
        if kres is None:
            st.warning("오늘 장 데이터가 없어요 (휴장일이거나 개장 전).")
        else:
            full = kres[kres["통과수"] == N_RULES].head(KR["max_cards"])
            st.subheader(f"{N_RULES}개 규칙 전부 통과 {len(full)}개 · 데이터 기준 {kasof} KST")
            if len(full) and st.button("⏱ 1분봉 다시 확인", use_container_width=True, key="kr_min"):
                with st.spinner("1분봉 확인 중..."):
                    mins, masof = kr_minute_check(full.index)
                store()[kr_key] = (kres, kasof, mins, masof)
            if masof:
                st.caption(f"1분봉 기준 {masof} KST")

            picks = []
            for sym, r in full.iterrows():
                mres = mins.get(sym, "– 1분봉 미확인")
                with st.container(border=True):
                    st.markdown(f"**{sym}** · ${r['현재가']:.4f} "
                                f"<span style='color:#94a3b8'>({r['현재가'] * fx:,.0f}원)</span> · "
                                f"당일 {pct(r['당일등락'])}", unsafe_allow_html=True)
                    st.caption(f"거래량 {r['거래량배수']:.1f}배 · 전고점 대비 {pct(r['고점대비'])} · "
                               f"위꼬리 {r['위꼬리'] * 100:.0f}% · 조정 {r['조정일수']}일 · "
                               f"저점높임 {'✅' if r['저점높임'] else '–'}")
                    stop = r["저가"]
                    st.markdown(f"🎯 목표가 **${r['현재가'] * 1.05:.4f}** (+5%) · "
                                f"🛑 손절가 **${stop:.4f}** (오늘 저가, {pct(stop / r['현재가'] - 1)})")
                    st.markdown(f"1분봉: {mres}")
                    c1, c2 = st.columns(2)
                    c1.markdown(f"[📰 뉴스 보기](https://finviz.com/quote.ashx?t={sym})")
                    if c2.checkbox("재료 확인함", key=f"kr_mat_{sym}"):
                        if mres.startswith("✅"):
                            picks.append(sym)

            st.divider()
            if picks:
                st.success(f"🎯 매수 후보: {', '.join(picks)}  (재료 확인 + 1분봉 ✅)")
            else:
                st.info("매수 후보 없음 — 재료 확인 + 1분봉 ✅ 둘 다 된 종목만 여기 떠요.")
            st.caption("👉 들어가기 전에 본좌로 희석(ATM·워런트·공모) 확인 · 보통 3~4종목, 빚투 금지")

            near = kres[kres["통과수"] == N_RULES - 1]
            if len(near):
                with st.expander(f"1개 부족 {len(near)}개"):
                    show = near[["현재가", "당일등락", "거래량배수", "고점대비", "조정일수", "탈락"]].copy()
                    show["당일등락"] = show["당일등락"].map(pct)
                    show["고점대비"] = show["고점대비"].map(pct)
                    st.dataframe(show.round(2), use_container_width=True)

    st.divider()
    st.subheader("📊 성적표")
    st.caption("과거에 이 탭이 잡았을 종목을 일봉으로 다시 계산해서 다음날 결과를 보여줘요. "
               "재료·1분봉 조건은 빠져 있고, 매수가는 그날 종가예요.")
    hc1, hc2 = st.columns([2, 1])
    period = hc1.radio("기간", [20, 60, 120], horizontal=True, format_func=lambda x: f"최근 {x}거래일",
                       key="kr_period", label_visibility="collapsed")
    hkey = f"krh_{today_key}_{period}_{KR['min_price']}_{up_pct}"
    if hc2.button("📊 성적 보기", use_container_width=True, key="kr_hist"):
        with st.spinner("계산 중..."):
            store()[hkey] = kr_history(period)
    hist = store().get(hkey)
    if hist is not None:
        done = hist.dropna(subset=["다음날시가"]) if len(hist) else hist
        if not len(done):
            st.info("이 기간엔 걸린 종목이 없어요.")
        else:
            a, b, c, d = st.columns(4)
            a.metric("검출", f"{len(done)}건")
            b.metric("다음날 시가 상승", pct((done["다음날시가"] > 0).mean()),
                     f"평균 {done['다음날시가'].mean() * 100:+.1f}%")
            c.metric("고가 +5% 도달", pct((done["다음날고가"] >= 0.05).mean()))
            d.metric("고가 +10% 도달", pct((done["다음날고가"] >= 0.10).mean()))
            st.markdown("**매도 방식별 결과** (수수료 전)")
            st.dataframe(kr_exit_table(done), hide_index=True, use_container_width=True)
            st.markdown("**+5% 잘 닿는 조건 찾기**")
            st.caption("도달률 높고 '못 닿을 때 평균'이 작은 구간이 좋은 구간. 건수 10건 미만은 믿지 말기")
            for tbl_name, tbl in kr_bucket_tables(done).items():
                st.markdown(f"<span style='font-size:13px;color:#94a3b8'>{tbl_name}</span>",
                            unsafe_allow_html=True)
                st.dataframe(tbl, hide_index=True, use_container_width=True)
            st.caption("손절선 = 매수한 날의 저가. 같은 날 +5%와 손절선을 둘 다 찍었으면 손절이 먼저라고 보수적으로 계산")
            st.caption(f"시가에 팔았을 때 중앙값 {done['다음날시가'].median() * 100:+.1f}% · "
                       f"종가까지 들고 있었으면 평균 {done['다음날종가'].mean() * 100:+.1f}% (수수료 전)")
            show = hist.drop(columns=["다음날저가", "손절선", "전일종가"]).sort_values("날짜", ascending=False)
            for col in ("다음날시가", "다음날고가", "다음날종가"):
                show[col] = show[col].map(lambda x: f"{x * 100:+.1f}%" if pd.notna(x) else "대기")
            show["거래량배수"] = show["거래량배수"].round(1)
            show["매수가"] = show["매수가"].round(2)
            st.dataframe(show, hide_index=True, use_container_width=True)

    with st.expander("규칙"):
        st.markdown(
            "**종목 선정 (전부 통과해야 카드)**\n"
            "1. 거래량: 20일 평균의 3배 이상 30배 미만 (거래대금 $5M 이상, 가격 $5~20 · 토글 켜면 $1~20)\n"
            "2. 전고점 근접: 종가가 60일 고점의 -10% 이내\n"
            "3. 깔끔한 양봉: 양봉 + 위꼬리가 캔들 길이의 25% 이하\n"
            "4. 기간 조정: 60일 고점이 5~30거래일 전\n"
            "5. 5일선 위\n"
            "6. 당일 상승률이 위에서 고른 하한 이상 (스팩·ETF·대형주 거르기)\n"
            "- 저점 높임(최근 5일 저점 > 그 전 5일 저점)은 선호 표시만\n\n"
            "**매수**: 1분봉에서 당일 저점 안 깨고 20선 위 우상향 확인 후, 장중 미리 사지 말 것. "
            "판단 안 서면 애프터에서 오르는지 보고 결정. 재료 이유 모르면 패스\n\n"
            "**매도 (성적표 ③번)**: 매수 직후 목표가(+5%) 지정가 매도 걸기. "
            "다음날 시가가 이미 +5% 넘으면 시가에 매도. 손절가(매수일 저가) 깨면 바로 손절, "
            "둘 다 안 닿으면 다음날 종가 전 정리. 물타기 금지")
