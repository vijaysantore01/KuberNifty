import os, sys, time, math, signal
import datetime as dt
import zoneinfo
import pandas as pd
import pyotp, requests, gzip, json
from SmartApi.smartConnect import SmartConnect
from twilio.rest import Client
from datetime import date
from flask import jsonify
from flask import Flask, render_template


# ---- CONFIG ----
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "<YOUR_TWILIO_SID>")
TWILIO_AUTH_TOKEN  = os.getenv("TWILIO_AUTH_TOKEN", "<YOUR_TWILIO_TOKEN>")
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
SMARTAPI_API_KEY = os.getenv("SMARTAPI_HIST_KEY", "2xmpcHx9")
hist_client = SmartConnect(api_key=SMARTAPI_API_KEY)
ANGEL_USER_ID  = os.getenv("ANGEL_ID", "AAAL213260")
ANGEL_PASSWORD = os.getenv("ANGEL_PIN", "0106")
TOTP_SEED      = os.getenv("ANGEL_TOTP_SEED")
if not TOTP_SEED:
    raise RuntimeError("Set ANGEL_TOTP_SEED env-var with your Base-32 secret")
UPSTOX_TOKEN_FILE = "upstox_v2_token.txt"
try:
    UPSTOX_TOKEN = open(UPSTOX_TOKEN_FILE).read().strip()
except FileNotFoundError:
    print(f"ERROR: Missing {UPSTOX_TOKEN_FILE}", file=sys.stderr)
    sys.exit(1)
UPSTOX_HEADERS = {"Accept":"application/json", "Authorization":f"Bearer {UPSTOX_TOKEN}"}
UPSTOX_BASE    = "https://api.upstox.com/v2"
SYMBOL           = "NIFTY"
ATR_LOW_THRESHOLD  = 25.0
TARGET_PCT = 0.20
SL_PCT     = 0.20
active_trade       = None
real_signal_history = []
UPS_FRONT_KEY      = None

# ---- ALL HELPERS ----
suggestion_history  = {}
real_signal_history = []

class TimeoutException(Exception): pass
def _timeout_handler(signum, frame): raise TimeoutException()

def fresh_login():
    otp = pyotp.TOTP(TOTP_SEED).now()
    hist_client.generateSession(ANGEL_USER_ID, ANGEL_PASSWORD, otp)
    print("✅ SmartAPI session (re)validated.")

def colorize(text: str, sentiment: str) -> str:
    s = sentiment.lower()
    if "bullish" in s or "📈" in s:
        return f"\033[92m{text}\033[0m"   # Green
    elif "bearish" in s or "📉" in s:
        return f"\033[91m{text}\033[0m"   # Red
    else:
        return f"\033[93m{text}\033[0m"   # Yellow


fresh_login()
hist_client.setSessionExpiryHook(lambda: fresh_login())
def fetch_historical_5m_with_market_check(upstox_key: str, minutes_back: int = 75) -> pd.DataFrame:
    """
    Fetch today’s 1-minute intraday bars from Upstox, resample into 5-minute bars,
    and return the last `minutes_back` of those 5-minute bars.
    If market is closed, returns an empty DataFrame.
    """
    if not is_market_open_ist():
        print("⚠️  Market is closed (outside 09:15–15:30 IST). Skipping historical fetch.\n")
        return pd.DataFrame()

    # 1) Pull today's intraday 1-minute candles
    url_intraday = f"{UPSTOX_BASE}/historical-candle/intraday/{upstox_key}/1minute"
    resp = requests.get(url_intraday, headers=UPSTOX_HEADERS, timeout=10)
    if resp.status_code != 200:
        print(f"⚠️ Intraday call failed (HTTP {resp.status_code})\n{resp.text}\n")
        return pd.DataFrame()

    intr_json = resp.json()
    # Expected format:
    # { "status":"success", "data": { "candles":[ [ "2025-06-04T09:15:00", open, high, low, close, volume, ... ], ... ] } }
    try:
        candles_1m = intr_json["data"]["candles"]
    except Exception:
        print("⚠️ Unexpected intraday JSON structure:", intr_json)
        return pd.DataFrame()

    # 2) Build a DataFrame of 1-minute bars. Upstox may return 6 or 7 columns per row.
    df_temp = pd.DataFrame(candles_1m)
    # Take only the first six columns (time, open, high, low, close, volume)
    df_1m = df_temp.iloc[:, :6]
    df_1m.columns = ["time_str", "open", "high", "low", "close", "volume"]
    df_1m["time"] = pd.to_datetime(df_1m["time_str"])
    df_1m.set_index("time", inplace=True)
    df_1m[["open","high","low","close"]] = df_1m[["open","high","low","close"]].astype(float)
    df_1m["volume"] = df_1m["volume"].astype(int)

    # 3) Resample into 5-minute bars
    df_5m = df_1m.resample("5min").agg({
        "open":  "first",
        "high":  "max",
        "low":   "min",
        "close": "last",
        "volume":"sum"
    }).dropna()

    # 4) Return only the last `minutes_back` bars
    if len(df_5m) < minutes_back:
        return df_5m
    return df_5m.tail(minutes_back)
def get_nifty_symboltoken():
    # Load instrument master if not already loaded
    df = pd.read_csv('NSE_full.csv')  # Or load from AngelOne-provided file
    row = df[df['tradingsymbol'] == 'NIFTY'].iloc[0]
    return str(row['exchange_token'])

def get_nifty_fut_symboltoken():
    df = pd.read_csv('complete.csv')
    df_fut = df[
        (df['tradingsymbol'].str.contains('NIFTY', case=False, na=False)) &
        (df['exchange'] == 'NSE_FO') &
        (df['instrument_type'] == 'FUTIDX')
    ]
    if df_fut.empty:
        print("DEBUG: No NIFTY FUTIDX found in complete.csv!")
        print(df.head())
        raise ValueError("NIFTY FUTIDX not found in master file")
    # Pick the one with earliest expiry
    df_fut['expiry_date'] = pd.to_datetime(df_fut['expiry'], errors='coerce')
    df_fut = df_fut.sort_values('expiry_date')
    return df_fut.iloc[0]['instrument_key']

def get_nifty_fut_symbol():
    df = pd.read_csv('complete.csv')
    # All current/future NIFTY futures contracts
    df_fut = df[
        (df['tradingsymbol'].str.contains('NIFTY', case=False, na=False)) &
        (df['exchange'] == 'NSE_FO') &
        (df['instrument_type'] == 'FUTIDX')
    ].copy()
    if df_fut.empty:
        print("DEBUG: No row for NIFTY/NSE_FO/FUTIDX found!")
        print("Available tradingsymbols:", df['tradingsymbol'].unique())
        print("Available exchanges:", df['exchange'].unique())
        print("Available instrument_types:", df['instrument_type'].unique())
        raise ValueError("NIFTY FUTIDX not found in master file")
    # Remove the SettingWithCopyWarning
    df_fut.loc[:, 'expiry_date'] = pd.to_datetime(df_fut['expiry'], errors='coerce')
    df_fut = df_fut.sort_values('expiry_date')
    row = df_fut.iloc[0]
    return row['tradingsymbol'], str(row['instrument_key'])



def download_upstox_master():
    url = "https://assets.upstox.com/market-quote/instruments/exchange/complete.json.gz"
    print("🔄  Downloading and decompressing Upstox instrument master …")
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    decompressed = gzip.decompress(resp.content)
    instruments = json.loads(decompressed)
    print(f"   → Loaded {len(instruments):,} total instruments.\n")
    return instruments

def auto_detect_upstox_fields(sample: dict):
    candidate_map = {
        "instrument_key": ["instrumentKey", "instrument_key", "instrumentToken", "instrument_token"],
        "segment":        ["segment", "exchange"],
        "instr_type":     ["instrumentType", "instrument_type", "type"],
        "expiry":         ["expiry", "expiryDate", "expiry_date"],
    }
    mapping = {}
    for logical, keys in candidate_map.items():
        found = None
        for k in keys:
            if k in sample:
                found = k
                break
        mapping[logical] = found
    return mapping

def get_sentiment_class(lbl):
    s = lbl.lower()
    if "bull" in s or "📈" in s:
        return "bullish"
    elif "bear" in s or "📉" in s:
        return "bearish"
    else:
        return "neutral"


def is_market_open_ist() -> bool:
    return True
    ist = zoneinfo.ZoneInfo("Asia/Kolkata")
    now = dt.datetime.now(ist)
    if now.weekday() >= 5:
        return False
    return dt.time(9,15) <= now.time() <= dt.time(15,30)

def detect_upstox_underlying(instruments: list, fields: dict):
    und_sym = None
    und_key = None
    seg_k      = fields["segment"]
    type_k     = fields["instr_type"]
    for inst in instruments:
        if inst.get(seg_k) == "NSE_FO" and inst.get(type_k) == "FUT":
            for cand in ["underlying_symbol", "underlyingSymbol", "underlying"]:
                if cand in inst:
                    und_sym = cand
                    break
            if "underlying_key" in inst:
                und_key = "underlying_key"
            break
    return und_sym, und_key

def find_front_month_nifty_fut(instruments: list, fields: dict, und_sym_k: str, und_key_k: str):
    df = pd.DataFrame(instruments)
    inst_key_k = fields["instrument_key"]
    seg_k      = fields["segment"]
    type_k     = fields["instr_type"]
    exp_k      = fields["expiry"]
    missing = [k for k in (inst_key_k, seg_k, type_k, exp_k) if k is None]
    if missing:
        raise RuntimeError(f"Missing JSON keys for: {missing}. Inspect the dump.")
    mask = (df[seg_k] == "NSE_FO") & (df[type_k] == "FUT")
    if und_sym_k:
        mask &= (df[und_sym_k] == "NIFTY")
    elif und_key_k:
        mask &= df[und_key_k].str.contains("Nifty", case=False, na=False)
    else:
        return None, None
    df_fut = df[mask].copy()
    if df_fut.empty:
        return None, None
    def to_date(v):
        if v is None:
            return pd.NaT
        try:
            return pd.to_datetime(int(v), unit="ms").date()
        except:
            try:
                return pd.to_datetime(v).date()
            except:
                return pd.NaT
    df_fut["expiry_date"] = df_fut[exp_k].apply(to_date)
    today = date.today()
    df_fut = df_fut[df_fut["expiry_date"] >= today]
    if df_fut.empty:
        return None, None
    df_fut.sort_values("expiry_date", inplace=True)
    front = df_fut.iloc[0]
    raw_key = front[inst_key_k]
    url_key = raw_key.replace("|", "%7C")
    return front, url_key

def get_spot_price(sym):
    try:
        #r = hist_client.ltpData(tradingsymbol=sym, symboltoken="40354")
        r = hist_client.ltpData(exchange="NSE", tradingsymbol=sym, symboltoken="40354")

        return float(r["data"]["last_price"])
    except Exception as e:
        print(f"LTP fetch failed (spot): {e}")
        # Try nsepython fallback
        try:
            oc = __import__('nsepython').nse_optionchain_scrapper(sym)
            return oc["records"]["underlyingValue"]
        except Exception as e2:
            print(f"NSEPython fallback failed: {e2}")
            # Special marker for market closed
            return None


def get_nifty_fut_ltp():
    df = pd.read_csv("complete.csv")
    # get nearest future contract for NIFTY (you may need to improve this part)
    fut_row = df[
        (df["tradingsymbol"].str.startswith("NIFTY")) &
        (df["exchange"] == "NSE_FO") &
        (df["instrument_type"] == "FUTIDX")
    ].sort_values("expiry").iloc[0]
    fut_sym = fut_row["tradingsymbol"]
    fut_token = str(fut_row["exchange_token"])  # Should be numeric string
    try:
        resp = hist_client.ltpData(tradingsymbol=fut_sym, symboltoken=fut_token, exchange="NSE")
        return float(resp["data"]["last_price"])
    except Exception as e:
        print("NIFTY FUT fetch failed:", e)
        return 0.0



def get_nearest_expiry(sym):
    #signal.signal(signal.SIGALRM, _timeout_handler)
    #signal.alarm(5)
    try:
        oc = __import__('nsepython').nse_optionchain_scrapper(sym)
        #signal.alarm(0)
        return oc["records"]["expiryDates"][0]
    except Exception:
        #signal.alarm(0)
        return None

def get_oi_data(sym, expiry):
    # Fetch option chain data
    oc = __import__('nsepython').nse_optionchain_scrapper(sym)
    rows = [r for r in oc["records"]["data"] if r["expiryDate"] == expiry]
    if not rows:
        return pd.DataFrame()  # Empty dataframe if nothing found

    df = pd.DataFrame(rows)
    # Parse OI & OI Change
    for leg in ("CE", "PE"):
        df[f"{leg}_OI"] = df[leg].map(lambda x: x.get("openInterest", 0) if isinstance(x, dict) else 0)
        df[f"{leg}_Change_OI"] = df[leg].map(
            lambda x: x.get("changeinOpenInterest", 0) if isinstance(x, dict) else 0)
    # Get symboltokens for LTP and IV
    toks = [x.get("symboltoken") for r in rows for x in (r["CE"], r["PE"]) if isinstance(x, dict)]
    lmap = {}
    for t in set(toks):
        try:
            #resp = hist_client.ltpData(tradingsymbol="", symboltoken=t)
            symboltoken = get_nifty_fut_symboltoken()
            resp = hist_client.ltpData(exchange="NSE", tradingsymbol=sym, symboltoken=symboltoken)
            d = resp["data"].get(t, resp["data"])
            lmap[t] = {
                "last_price": d.get("last_price", 0.0),
                "implied_volatility": d.get("implied_volatility", 0.0)
            }
        except Exception:
            lmap[t] = {}

    def ltp(x):
        return float(lmap.get(x.get("symboltoken"), {}).get("last_price", x.get("lastPrice", 0.0))) if isinstance(x,
                                                                                                                  dict) else 0.0

    def iv(x):
        return float(lmap.get(x.get("symboltoken"), {}).get("implied_volatility",
                                                            x.get("impliedVolatility", 0.0))) if isinstance(x,
                                                                                                            dict) else 0.0

    df["CE_LTP"] = df["CE"].map(ltp)
    df["PE_LTP"] = df["PE"].map(ltp)
    df["CE_IV"] = df["CE"].map(iv)
    df["PE_IV"] = df["PE"].map(iv)

    # Rename and select columns
    df = df.rename(columns={"strikePrice": "Strike"})
    df = df[["Strike", "CE_OI", "PE_OI", "CE_Change_OI", "PE_Change_OI", "CE_LTP", "PE_LTP", "CE_IV", "PE_IV"]]

    # Compute bias
    df["Bias"] = df.apply(
        lambda r: "Bullish" if r.PE_Change_OI > r.CE_Change_OI
        else ("Bearish" if r.CE_Change_OI > r.PE_Change_OI else "Neutral"),
        axis=1
    )

    return df.sort_values("Strike").reset_index(drop=True)

def filter_strikes_near_spot(df, spot, window=5, step=50):
    atm = min(df.Strike, key=lambda x: abs(x - spot))
    low, high = atm - window * step, atm + window * step
    return df[df.Strike.between(low, high)].copy(), atm

def calculate_pcr(df):
    return round(df.PE_OI.sum() / df.CE_OI.sum(), 2) if df.CE_OI.sum() > 0 else 0.0

def calculate_option_skew(df, spot):
    idx = (df.Strike - spot).abs().idxmin()
    r = df.loc[idx]
    return round(r.CE_LTP - r.PE_LTP, 2), r.CE_LTP, r.PE_LTP, r.Strike

def calculate_max_pain(df):
    strikes = df.Strike.values
    losses = [((s - strikes).clip(0) * df.PE_OI).sum() + ((strikes - s).clip(0) * df.CE_OI).sum() for s in strikes]
    mi = int(pd.Series(losses).idxmin())
    return int(strikes[mi]), losses[mi]

def summarize_sentiment(df):
    cnt = df.Bias.value_counts()
    b, r = cnt.get("Bullish", 0), cnt.get("Bearish", 0)
    lbl = "📈 Bullish" if b > r else ("📉 Bearish" if r > b else "⚖️ Neutral")
    return lbl, b, r

def days_to_expiry(exp):
    for fmt in ("%Y-%m-%d", "%d-%b-%Y"):
        try:
            d = dt.datetime.strptime(exp, fmt).date()
            return max((d - dt.date.today()).days, 0)
        except:
            pass
    return 0

def theta_call_per_year(S, K, r, sig, T):
    if T <= 0 or sig <= 0: return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    npdf = math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)
    Ncdf = 0.5 * (1 + math.erf(d2 / math.sqrt(2)))
    return -S * sig * npdf / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * Ncdf

def breakdown_confidence_extended(
        pcr, skew, mp, spot, bcnt, rcnt, prev_close, today_high, today_low, ma_val, vol_bias, skip_price,
        aggressive=False):
    score, steps = 0, []
    if aggressive:
        if pcr > 1.05:
            score += 1
        elif pcr < 0.95:
            score -= 1
    else:
        if pcr > 1.10:
            score += 1
        elif pcr < 0.80:
            score -= 1
    if spot < mp:
        score += 1
    elif spot > mp:
        score -= 1
    if bcnt > rcnt:
        score += 1
    elif rcnt > bcnt:
        score -= 1
    oi_only = score
    if oi_only >= 2:     return +4, steps
    if oi_only <= -2:    return -4, steps
    if skip_price:       return score, steps
    return score, steps

def final_suggestion_extended(score, skip_price):
    if skip_price:
        conf = int((abs(score) / 4) * 100)
        if score >= 2:
            return "BUY CALL", conf
        elif score <= -2:
            return "BUY PUT", conf
        else:
            return "Stay Neutral", conf
    else:
        conf = int((abs(score) / 8) * 100)
        if score >= 4:
            return "BUY CALL", conf
        elif score <= -4:
            return "BUY PUT", conf
        else:
            return "Stay Neutral", conf

def generate_final_suggestion(
    pcr, skew_val, spot, mp, atr_val, ma_15m, ATR_LOW_THRESHOLD=25.0
):
    """
    Returns: suggestion (str), final_score (int), oi_score (int), price_score (int)
    """
    oi_score = 0
    if pcr < 0.90:        oi_score -= 1
    elif pcr > 1.10:      oi_score += 1

    if skew_val < -3:     oi_score -= 1
    elif skew_val > +3:   oi_score += 1

    if spot > mp + 20:    oi_score -= 1
    elif spot < mp - 20:  oi_score += 1

    price_score = 0
    if spot < ma_15m:     price_score -= 1
    elif spot > ma_15m:   price_score += 1

    # Only act if ATR is sufficient
    if atr_val > ATR_LOW_THRESHOLD:
        final_score = oi_score + price_score
    else:
        final_score = 0  # No trade if volatility is low

    # Decision logic
    if final_score <= -2:
        suggestion = "BUY PUT"
    elif final_score >= 2:
        suggestion = "BUY CALL"
    else:
        suggestion = "Stay Neutral"

    return suggestion, final_score, oi_score, price_score

def get_live_premium(df, atm, signal):
    if signal == "BUY CALL":
        try:
            return df.loc[df["Strike"] == atm, "CE_LTP"].iloc[0]
        except:
            return None
    if signal == "BUY PUT":
        try:
            return df.loc[df["Strike"] == atm, "PE_LTP"].iloc[0]
        except:
            return None
    return None

def print_compact_dashboard(
    df: pd.DataFrame,
    spot: float,
    real_history: list,
    AGGR: bool = True,
    atm=None,
    ce_ltp=None,
    pe_ltp=None,
    ups_front_key=None,
    ATR_LOW_THRESHOLD=25.0
):
    import math
    import zoneinfo
    import datetime as dt


    now_ist = dt.datetime.now(zoneinfo.ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S IST")
    expiry = get_nearest_expiry("NIFTY") or "N/A"

    # OI
    pcr = calculate_pcr(df)
    skew_val, ce_ltp, pe_ltp, atm = calculate_option_skew(df, spot)
    mp, mp_loss = calculate_max_pain(df)
    label_oi, bc, rc = summarize_sentiment(df)

    # Upstox/Price-Action
    skip_price = False
    atr_val = today_high = today_low = ma_15m = 0.0
    vol_bias = 0
    prev_close = spot
    ATR_MSG = ""

    if ups_front_key is None:
        skip_price = True
        ATR_MSG = "⚠️  Upstox FUT not found; skipping price action"
    else:
        df_bars = fetch_historical_5m_with_market_check(ups_front_key, minutes_back=75)
        if df_bars.empty:
            skip_price = True
            ATR_MSG = "⚠️  No bars (market closed or no data); skipping price action"
        else:
            highs = df_bars["high"].tolist()
            lows = df_bars["low"].tolist()
            closes = df_bars["close"].tolist()
            prev_close = closes[-15] if len(closes) >= 15 else closes[0]
            if len(df_bars) < 15:
                skip_price = True
                ATR_MSG = f"⚠️  Only {len(df_bars)} bars; skipping ATR"
            else:
                tr_list = []
                tr0 = max(highs[0] - lows[0], abs(highs[0] - prev_close), abs(lows[0] - prev_close))
                tr_list.append(tr0)
                for i in range(1, 15):
                    tr = max(
                        highs[i] - lows[i],
                        abs(highs[i] - closes[i-1]),
                        abs(lows[i] - closes[i-1])
                    )
                    tr_list.append(tr)
                atr_val = sum(tr_list) / len(tr_list)
                today_high = max(highs[-15:])
                today_low = min(lows[-15:])
                ma_15m = sum(closes[-3:]) / 3 if len(closes) >= 3 else closes[-1]
                volumes = df_bars["volume"].tolist()
                if len(volumes) >= 2:
                    last_vol = volumes[-1]
                    avg_prev = sum(volumes[:-1]) / len(volumes[:-1])
                    threshold = 1.2 if AGGR else 1.5
                    if last_vol > avg_prev * threshold:
                        last_bar = df_bars.iloc[-1]
                        vol_bias = 1 if last_bar["close"] > last_bar["open"] else (-1 if last_bar["close"] < last_bar["open"] else 0)
                    else:
                        vol_bias = 0

    # Scores (use your scoring function here!)
    score_oi, _ = breakdown_confidence_extended(pcr, skew_val, mp, spot, bc, rc, spot, spot, spot, spot, 0, True, aggressive=True)
    score_price, _ = breakdown_confidence_extended(pcr, skew_val, mp, spot, bc, rc, prev_close, today_high, today_low, ma_15m, vol_bias, skip_price, aggressive=False)
    score_real, _ = breakdown_confidence_extended(pcr, skew_val, mp, spot, bc, rc, prev_close, today_high, today_low, ma_15m, vol_bias, skip_price, aggressive=AGGR)

    sig_oi, conf_oi = final_suggestion_extended(score_oi, True)
    sig_real, conf_real = final_suggestion_extended(score_real, skip_price)

    sentiment_market = "Bullish" if score_real > 0 else ("Bearish" if score_real < 0 else "Neutral")

    # Greeks
    try:
        iv_atm = (df.loc[df["Strike"] == atm, "CE_IV"].iloc[0] + df.loc[df["Strike"] == atm, "PE_IV"].iloc[0]) / 200.0
    except:
        iv_atm = 0.01
    rd = days_to_expiry(expiry)
    Tyrs = rd / 365.0
    r = 0.06
    if Tyrs > 0 and iv_atm > 0:
        d1 = (math.log(spot / atm) + (r + 0.5 * iv_atm * iv_atm) * Tyrs) / (iv_atm * math.sqrt(Tyrs))
        delta_c = 0.5 * (1 + math.erf(d1 / math.sqrt(2)))
        delta_p = delta_c - 1
    else:
        delta_c, delta_p = 0.0, 0.0
    theta_y = theta_call_per_year(spot, atm, r, iv_atm, Tyrs)
    theta_d = theta_y / 365.0
    tot_th = theta_d * rd

    # Signal delta
    premium_real = ce_ltp if sig_real == "BUY CALL" else (pe_ltp if sig_real == "BUY PUT" else None)
    delta_str = ""
    if premium_real is not None:
        key = f"{sig_real}@{atm}"
        if real_history and isinstance(real_history[-1], (list, tuple)) and real_history[-1][3] == sig_real and real_history[-1][4] == atm:
            d_real = premium_real - real_history[-1][2]
        else:
            d_real = 0.0
        delta_str = f" | ΔPrem: {d_real:+.2f}"

    output = []
    output.append("─────────────────────────────────────────────────────────────────────────────")
    output.append("\033[93m             ॐ श्री कुबेराय नमः - शुभ लाभ, शुभ व्यापार!\033[0m")
    #output.append(colorize(f"Market Direction: {label_oi}", sentiment_market))
    output.append(colorize(f"Market Direction: {label_oi}", label_oi))

    output.append(f"Time: {now_ist}  │  Spot: ₹{spot:,.2f}  (Δ {spot-prev_close:+.2f}, {((spot-prev_close)/prev_close*100 if prev_close else 0):+.2f}%)  │  Expiry: {expiry}")
    output.append("─────────────────────────────────────────────────────────────────────────────")
    output.append("A) OI Metrics:")
    count_bias = "Bullish" if bc > rc else ("Bearish" if rc > bc else "Neutral")
    output.append(f"   • StrikeCount: {colorize(f'Bull {bc} | Bear {rc} (Δ {bc-rc:+})', count_bias)}")
    pcr_label = "Bullish" if (pcr > (1.05 if AGGR else 1.10)) else ("Bearish" if (pcr < (0.95 if AGGR else 0.80)) else "Neutral")
    output.append(f"   • PCR = {pcr:.2f}  ({colorize(pcr_label, pcr_label)})")
    skew_label = "Bullish" if (skew_val > (0.5 if AGGR else 0.0)) else ("Bearish" if (skew_val < (-0.5 if AGGR else 0.0)) else "Neutral")
    output.append(f"   • Skew = {skew_val:+.2f}  ({colorize(skew_label, skew_label)})")
    pain_label = "Bullish" if spot < mp else ("Bearish" if spot > mp else "Neutral")
    comp = "<" if spot < mp else (">" if spot > mp else "=")
    output.append(f"   • Spot vs MaxPain = ₹{spot:.2f} {comp} ₹{mp:.2f}  ({colorize(pain_label, pain_label)})")
    output.append(f"   • MaxPain: {mp}  | Loss: ₹{mp_loss:,.0f}")
    output.append("")

    output.append("B) Price-Action:")
    if skip_price:
        output.append(f"   • {ATR_MSG}")
    else:
        output.append(f"   • ATR(70m) = {atr_val:.2f}  (Thr={ATR_LOW_THRESHOLD} → {colorize('PASS' if atr_val>=ATR_LOW_THRESHOLD else 'FAIL', 'Bullish' if atr_val>=ATR_LOW_THRESHOLD else 'Bearish')})")
        output.append(f"   • PrevClose = ₹{prev_close:.2f}  |  Day H/L = ₹{today_high:.2f} / ₹{today_low:.2f}")
        output.append(f"   • 15m-MA = ₹{ma_15m:.2f}  |  VolBias = {colorize('Bullish' if vol_bias>0 else 'Bearish' if vol_bias<0 else 'Neutral', 'Bullish' if vol_bias>0 else 'Bearish' if vol_bias<0 else 'Neutral')}")
        sub_spc = f"{'+1' if spot>prev_close else '-1' if spot<prev_close else '0'}"
        sub_hl  = f"{'+1' if spot>=today_high else '-1' if spot<=today_low else '0'}"
        sub_ma  = f"{'+1' if spot>ma_15m else '-1' if spot<ma_15m else '0'}"
        sub_vol = f"{'+1' if vol_bias>0 else '-1' if vol_bias<0 else '0'}"
        output.append(f"   • Sub-scores: PrevClose {sub_spc}, H/L {sub_hl}, 15mMA {sub_ma}, Vol {sub_vol}")
    output.append("")

    output.append("C) Combined Scores:")
    output.append(f"   • OI-Only (4F) = {score_oi:+} / 4  → {conf_oi:>3d}%   {colorize(sig_oi, sig_oi)} @ {atm}")
    output.append(f"   • Price-Incl (8F) = {score_real:+} / 8  → {conf_real:>3d}%   {colorize(sig_real, sig_real)} @ {atm}{delta_str}")
    output.append("")

    output.append("D) Greeks & Decay:")
    output.append(f"   • IV(ATM avg) = {iv_atm*100:5.2f}%  |  Days→Exp = {rd}")
    output.append(f"   • Δ(Call) = {delta_c:+.2f}  |  Δ(Put) = {delta_p:+.2f}")
    output.append(f"   • Θ/day = {theta_d:+.2f}  |  TotalΘ = {tot_th:+.2f}")
    output.append("")

    output.append("E) Real Signal History (last 5):")
    output.append("   Time     │ Spot    │ Premium │ Signal     │ Strike │ ΔPrem")
    output.append("   -------------------------------------------------------------")
    last5 = real_history[-5:] if real_history else []
    def get_live_premium_for_row(row):
        signal, atm = row[3], row[4]
        if signal == "BUY CALL":
            try:
                return float(df.loc[df["Strike"] == atm, "CE_LTP"].iloc[0])
            except:
                return None
        elif signal == "BUY PUT":
            try:
                return float(df.loc[df["Strike"] == atm, "PE_LTP"].iloc[0])
            except:
                return None
        return None

    for ts, sp, pr, sg, strk in last5:
        live_prem = get_live_premium_for_row((ts, sp, pr, sg, strk))
        delta = live_prem - pr if (live_prem is not None and pr) else None
        delta_str_hist = f"{delta:+.2f}" if delta is not None else "--"
        output.append(f"   {ts:8} │ ₹{sp:7.2f} │ ₹{pr:7.2f} │ {colorize(sg, sg):9} │ @{strk:<5} │ {delta_str_hist:>6}")

    for _ in range(5 - len(last5)):
        output.append("   ")
    output.append("─────────────────────────────────────────────────────────────")
    print('\n'.join(output))



def log_and_remember(ts, spot, prem, sig, strike):
    real_signal_history.append((ts, spot, prem, sig, strike))

# ---- MAIN LOOP ----

# ---- Your config and helper code here (UNCHANGED) ----
# Copy ALL your helper functions and logic here!
# ... (omitted for brevity - paste your logic) ...

# Initialize API clients, set up all helpers, fresh_login, etc
# (Just as you did in your script)

# Flask app
app = Flask(__name__)

def get_dashboard_data():
    # --- Insert all your logic here ---
    spot = get_spot_price(SYMBOL)
    expiry = get_nearest_expiry(SYMBOL)
    df = get_oi_data(SYMBOL, expiry)
    if not is_market_open_ist():
        return {
            'market_sentiment': 'Market Closed',
            'sentiment_class': 'neutral',
            'spot': 'N/A',
            'max_pain': 'N/A',
            'pcr': 'N/A',
            'skew': 'N/A',
            'oi_score': 'N/A',
            'price_score': 'N/A',
            'final_signal': 'Market Closed',
            'signal_history': [],
            'market_closed': True
        }
    if spot is None or spot == 0:
        # Market closed or no data
        return {
            'market_closed': True,
            'market_sentiment': 'No Data',
            'sentiment_class': 'neutral',
            'spot': 'N/A',
            'max_pain': 'N/A',
            'pcr': 'N/A',
            'skew': 'N/A',
            'oi_score': 0,
            'price_score': 0,
            'final_signal': 'No Signal',
            'signal_history': [],
        }
    if df.empty:
        return {
            'market_closed': False,
            'market_sentiment': 'No Data',
            'sentiment_class': 'neutral',
            'spot': 0,
            'max_pain': 0,
            'pcr': 0,
            'skew': 0,
            'oi_score': 0,
            'price_score': 0,
            'final_signal': 'No Signal',
            'signal_history': [],
        }

    df_filt, atm = filter_strikes_near_spot(df, spot)
    pcr = calculate_pcr(df_filt)
    skew_val, ce_ltp, pe_ltp, atm = calculate_option_skew(df_filt, spot)
    mp, mp_loss = calculate_max_pain(df_filt)
    lbl, bc, rc = summarize_sentiment(df_filt)

    atr_val = 0.0
    ma_15m = spot
    prev_close = spot
    today_high = today_low = spot
    vol_bias = 0
    if UPS_FRONT_KEY:
        df_bars = fetch_historical_5m_with_market_check(UPS_FRONT_KEY, minutes_back=75)
        if not df_bars.empty and len(df_bars) >= 15:
            closes = df_bars["close"].tolist()
            highs = df_bars["high"].tolist()
            lows = df_bars["low"].tolist()
            prev_close = closes[-15] if len(closes) >= 15 else closes[0]
            today_high = max(highs[-15:])
            today_low = min(lows[-15:])
            ma_15m = sum(closes[-3:]) / 3 if len(closes) >= 3 else closes[-1]
            tr_list = []
            for i in range(15):
                tr = max(
                    highs[i] - lows[i],
                    abs(highs[i] - closes[i-1]) if i > 0 else 0,
                    abs(lows[i] - closes[i-1]) if i > 0 else 0
                )
                tr_list.append(tr)
            atr_val = sum(tr_list) / len(tr_list)
            # Volume bias logic
            volumes = df_bars["volume"].tolist()
            if len(volumes) >= 2:
                last_vol = volumes[-1]
                avg_prev = sum(volumes[:-1]) / len(volumes[:-1])
                if last_vol > avg_prev * 1.2:
                    last_bar = df_bars.iloc[-1]
                    vol_bias = 1 if last_bar["close"] > last_bar["open"] else (-1 if last_bar["close"] < last_bar["open"] else 0)

    suggestion, final_score, oi_score, price_score = generate_final_suggestion(
        pcr, skew_val, spot, mp, atr_val, ma_15m, ATR_LOW_THRESHOLD
    )

    # Build recent signal history (you may want to limit to last 5-10)
    signal_history = []
    for row in real_signal_history[-10:]:
        ts, sp, pr, sig, strk = row
        # Get live premium for delta if available
        live_prem = None
        try:
            if sig == "BUY CALL":
                live_prem = float(df_filt.loc[df_filt["Strike"] == strk, "CE_LTP"].iloc[0])
            elif sig == "BUY PUT":
                live_prem = float(df_filt.loc[df_filt["Strike"] == strk, "PE_LTP"].iloc[0])
        except:
            live_prem = None
        delta = (live_prem - pr) if (live_prem is not None and pr) else None
        signal_history.append([
            ts, f"₹{sp:.2f}", f"₹{pr:.2f}", sig, strk, (f"{delta:+.2f}" if delta is not None else "--")
        ])

    # Sentiment coloring
    sentiment_class = 'neutral'
    if "bull" in suggestion.lower():
        sentiment_class = 'bullish'
    elif "put" in suggestion.lower():
        sentiment_class = 'bearish'

    return {
        'market_closed': False,
        'market_sentiment': lbl,
        'sentiment_class': sentiment_class,
        'spot': f"{spot:,.2f}",
        'max_pain': mp,
        'pcr': pcr,
        'skew': skew_val,
        'oi_score': oi_score,
        'price_score': price_score,
        'final_signal': suggestion,
        'signal_history': signal_history
    }



@app.route("/api/dashboard")
def dashboard_api():
    dashboard = get_dashboard_data()
    return jsonify(dashboard)

@app.route("/")
def dashboard():

    dashboard = get_dashboard_data()  # your backend function!
    dashboard['sentiment_class'] = get_sentiment_class(dashboard['market_sentiment'])

    return render_template("dashboard.html", dashboard=dashboard)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

