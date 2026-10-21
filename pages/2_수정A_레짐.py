# ==========================================================
#  수정A 레짐 판정  (QQQ 추세 기준 TQQQ / QQQ / 현금)
#  - 규칙: QQQ가 100·150·200일선 중 2개 이상 위 → 상승추세
#          상승추세 + 12개월 수익률 플러스 → TQQQ
#          상승추세 + 12개월 수익률 마이너스 → QQQ
#          이동평균 2개 이상 아래 → 현금
#          새 판정이 2일 연속 같아야 교체, 교체는 다음 날 본장 시가
#  - 데이터: 알파카 일봉 (스트림릿 Secrets의 알파카 키 사용)
# ==========================================================
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import streamlit as st

# ---------- 설정 ----------
MAS = (100, 150, 200)
CONFIRM_DAYS = 2
DATA_START = "2016-01-01"
BT_START = "2018-01-01"
NAMES = {"T": "TQQQ", "Q": "QQQ", "": "현금"}
COLORS = {"T": "🟢", "Q": "🟡", "": "🔴"}

# 매크로 일정 (뉴욕 날짜) — 출처: 연준 FOMC 일정표, BLS 발표 일정표. 해가 바뀌면 추가
MACRO = [
    ("2026-10-28", "FOMC 결과"), ("2026-11-06", "고용보고서"), ("2026-11-10", "CPI"),
    ("2026-12-04", "고용보고서"), ("2026-12-09", "FOMC 결과"), ("2026-12-10", "CPI"),
    ("2027-01-08", "고용보고서"), ("2027-01-13", "CPI"), ("2027-01-27", "FOMC 결과"),
    ("2027-02-05", "고용보고서"), ("2027-02-11", "CPI"),
    ("2027-03-05", "고용보고서"), ("2027-03-10", "CPI"), ("2027-03-17", "FOMC 결과"),
    ("2027-04-02", "고용보고서"), ("2027-04-13", "CPI"), ("2027-04-28", "FOMC 결과"),
    ("2027-05-07", "고용보고서"), ("2027-05-12", "CPI"),
    ("2027-06-04", "고용보고서"), ("2027-06-09", "FOMC 결과"), ("2027-06-10", "CPI"),
    ("2027-07-02", "고용보고서"), ("2027-07-14", "CPI"), ("2027-07-28", "FOMC 결과"),
    ("2027-08-06", "고용보고서"), ("2027-08-11", "CPI"),
    ("2027-09-03", "고용보고서"), ("2027-09-14", "CPI"), ("2027-09-15", "FOMC 결과"),
    ("2027-10-08", "고용보고서"), ("2027-10-14", "CPI"), ("2027-10-27", "FOMC 결과"),
    ("2027-11-05", "고용보고서"), ("2027-11-10", "CPI"),
    ("2027-12-03", "고용보고서"), ("2027-12-08", "FOMC 결과"), ("2027-12-10", "CPI"),
]


# ========== 계산 (화면과 분리) ==========
def raw_signal(close: pd.Series) -> pd.Series:
    """그날 종가 기준 원판정: 'T' / 'Q' / '' / None(데이터 부족)"""
    above = sum((close > close.rolling(n).mean()).astype(int) for n in MAS)
    r12 = close / close.shift(252) - 1
    out = pd.Series(np.where(above >= 2, np.where(r12 > 0, "T", "Q"), ""), index=close.index, dtype=object)
    out[r12.isna() | close.rolling(max(MAS)).mean().isna()] = None
    return out


def confirm(raw: pd.Series, k: int = CONFIRM_DAYS) -> pd.Series:
    """새 판정이 k일 연속 같아야 바뀜"""
    vals, out, cur = list(raw.values), [], None
    for i, s in enumerate(vals):
        if s is None:
            out.append(None); continue
        if cur is None:
            cur = s
        elif s != cur and i - k + 1 >= 0 and all(vals[i - j] == s for j in range(k)):
            cur = s
        out.append(cur)
    return pd.Series(out, index=raw.index, dtype=object)


def backtest(sig: pd.Series, px: dict, start: str):
    """판정일 종가 → 다음 날 시가에 교체. px = {'T': df(open,close), 'Q': df(open,close)}"""
    days = sig.index
    eq, held, rows = 1.0, None, []
    for i in range(1, len(days)):
        d, p = days[i], days[i - 1]
        if str(d) < start or sig.iloc[i - 1] is None:
            continue
        want = sig.iloc[i - 1]
        if held is None:
            held = want
        r = 1.0
        if want != held:
            if held:
                r *= px[held].at[d, "open"] / px[held].at[p, "close"]
            if want:
                r *= px[want].at[d, "close"] / px[want].at[d, "open"]
            held = want
        elif held:
            r *= px[held].at[d, "close"] / px[held].at[p, "close"]
        eq *= r
        rows.append((d, eq))
    return pd.Series([e for _, e in rows], index=[d for d, _ in rows])


def summary(eq: pd.Series):
    yrs = len(eq) / 252
    return {
        "총수익률": f"{(eq.iloc[-1] - 1) * 100:,.0f}%",
        "연평균": f"{(eq.iloc[-1] ** (1 / yrs) - 1) * 100:.1f}%",
        "최대낙폭": f"{(eq / eq.cummax() - 1).min() * 100:.1f}%",
    }


# ========== 데이터 ==========
def _alpaca_keys():
    def flat(d, prefix=""):
        for k in d.keys():
            v, p = d[k], f"{prefix}.{k}".lower()
            if hasattr(v, "keys"):
                yield from flat(v, p)
            else:
                yield p, v
    items = list(flat(st.secrets))
    pick = lambda cond: next((v for p, v in items if cond(p)), None)
    al = lambda p: "alpaca" in p or "apca" in p
    key = pick(lambda p: al(p) and "secret" not in p and ("key" in p or "id" in p)) or \
          pick(lambda p: "secret" not in p and "key" in p)
    sec = pick(lambda p: al(p) and "secret" in p) or pick(lambda p: "secret" in p)
    return key, sec


@st.cache_data(ttl=1800, show_spinner="시세 받는 중...")
def load_daily(sym: str) -> pd.DataFrame:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import Adjustment, DataFeed

    key, sec = _alpaca_keys()
    client = StockHistoricalDataClient(key, sec)
    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    df = None
    for feed in (DataFeed.SIP, DataFeed.IEX):
        try:
            req = StockBarsRequest(symbol_or_symbols=sym, timeframe=TimeFrame.Day,
                                   start=pd.Timestamp(DATA_START, tz="UTC"), end=end,
                                   adjustment=Adjustment.ALL, feed=feed)
            df = client.get_stock_bars(req).df
            break
        except Exception:
            continue
    df = df.reset_index()
    df["date"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York").dt.date
    df = df.set_index("date")[["open", "close"]].astype(float)
    # 장중이면 오늘 봉(미완성)은 버림
    ny = pd.Timestamp.now(tz="America/New_York")
    if df.index[-1] == ny.date() and (ny.hour, ny.minute) < (16, 15):
        df = df.iloc[:-1]
    return df


# ========== 화면 ==========
def main():
    st.set_page_config(page_title="수정A 레짐", page_icon="🚦", layout="centered")
    st.title("🚦 수정A 레짐")

    try:
        q, t = load_daily("QQQ"), load_daily("TQQQ")
    except Exception as e:
        st.error(f"시세를 못 받았어요. 스트림릿 Secrets의 알파카 키를 확인해 주세요. ({e.__class__.__name__})")
        return
    days = q.index.intersection(t.index)
    q, t = q.loc[days], t.loc[days]
    close = q["close"]
    raw = raw_signal(close)
    sig = confirm(raw)

    last_day = days[-1]
    now_s, prev_s, raw_s = sig.iloc[-1], sig.iloc[-2], raw.iloc[-1]

    # 1) 신호등
    st.caption(f"판정 기준: 뉴욕 {last_day} 종가")
    st.markdown(f"## {COLORS[now_s]} 지금 들고 있을 것: **{NAMES[now_s]}**")
    if now_s != prev_s:
        st.warning(f"📢 **교체일** — 오늘 밤 본장 개장 후 **{NAMES[prev_s]} → {NAMES[now_s]}**")
    elif raw_s != now_s:
        st.info(f"⏳ 오늘 종가 판정은 {NAMES[raw_s]} — 다음 종가도 같으면 교체")
    else:
        st.success("유지 — 오늘은 할 일 없음")

    # 2) 판정 근거
    with st.expander("판정 근거", expanded=False):
        for n in MAS:
            ma = close.rolling(n).mean().iloc[-1]
            st.write(f"{'✅' if close.iloc[-1] > ma else '❌'} {n}일선 {ma:,.2f} (QQQ {close.iloc[-1]:,.2f})")
        r12 = close.iloc[-1] / close.iloc[-253] - 1
        st.write(f"{'✅' if r12 > 0 else '❌'} 12개월 수익률 {r12 * 100:+.1f}%")

    # 3) 연습용 수량 계산
    amt = st.number_input("운용 금액 ($)", min_value=0, value=100_000, step=10_000)
    if now_s:
        ref = (t if now_s == "T" else q)["close"].iloc[-1]
        st.write(f"참고가 ${ref:,.2f} (전일 종가) → **{NAMES[now_s]} {math.floor(amt / ref):,}주**")
        st.caption("개장 후 30분 지나서 현재가 근처 지정가 · 마감 1시간 전까지 안 사지면 현재가로 다시")
    else:
        st.write("현금 보유 (원하면 SGOV 같은 초단기채)")

    # 4) 매크로 일정
    st.subheader("📅 2주 안 매크로 일정")
    today = pd.Timestamp(last_day)
    upcoming = [(d, n) for d, n in MACRO if today < pd.Timestamp(d) <= today + pd.Timedelta(days=14)]
    if upcoming:
        for d, n in upcoming:
            st.write(f"- {d} {n}")
        if now_s == "T":
            st.caption("TQQQ 보유 중: 발표 전날 QQQ로 낮출지는 직접 판단")
    else:
        st.write("없음")
    if MACRO and pd.Timestamp(MACRO[-1][0]) < today + pd.Timedelta(days=30):
        st.caption("⚠️ 매크로 일정표 끝이 가까워요. 코드의 MACRO 목록에 다음 해 일정을 추가해 주세요")

    # 5) 교체 기록 + 성적표
    st.subheader("🔁 최근 교체")
    sw = sig[(sig != sig.shift(1)) & sig.shift(1).notna()]
    st.dataframe(pd.DataFrame({"판정일": [str(d) for d in sw.index[-8:][::-1]],
                               "→": [NAMES[s] for s in sw.values[-8:][::-1]]}),
                 hide_index=True, use_container_width=True)

    st.subheader(f"📊 성적표 ({BT_START[:4]}~, 다음 날 시가 체결)")
    px = {"T": t, "Q": q}
    eq = backtest(sig, px, BT_START)
    base = eq.index
    qh = (q.loc[base, "close"] / q["close"].shift(1).loc[base]).cumprod()
    th = (t.loc[base, "close"] / t["close"].shift(1).loc[base]).cumprod()
    st.dataframe(pd.DataFrame({"수정A": summary(eq), "QQQ 보유": summary(qh), "TQQQ 보유": summary(th)}),
                 use_container_width=True)
    import altair as alt
    chart_df = pd.DataFrame({"수정A": eq, "QQQ 보유": qh, "TQQQ 보유": th})
    chart_df.index = pd.to_datetime(chart_df.index)
    long = chart_df.reset_index(names="날짜").melt("날짜", var_name="전략", value_name="배수")
    st.altair_chart(alt.Chart(long).mark_line().encode(
        x="날짜:T", y=alt.Y("배수:Q", scale=alt.Scale(type="log")), color="전략:N"),
        use_container_width=True)
    st.caption("비용·세금 제외, 현금 이자 0 가정. 과거 성적은 미래를 보장하지 않음")


if __name__ == "__main__":
    main()
