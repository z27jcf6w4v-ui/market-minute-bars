from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"


def classify(g):
    o = float(g.iloc[0].open)
    c = float(g.iloc[-1].close)
    h = float(g.high.max())
    l = float(g.low.min())

    rng = max(h - l, 1e-9)
    location = (c - l) / rng
    oc = (c / o - 1) * 100

    if oc >= 1 and location >= 0.70:
        return "Trend Up"
    if oc <= -1 and location <= 0.30:
        return "Trend Down"
    if abs(oc) < 0.60:
        return "Range Day"

    return "Reversal/Mixed"


def metrics_for_symbol(g, prior_close=None):

    g = g.sort_values("timestamp").copy()

    typical_price = (
        g.high +
        g.low +
        g.close
    ) / 3

    volume = g.volume.fillna(0)

    g["vwap"] = (
        (typical_price * volume).cumsum()
        / volume.cumsum().replace(0, np.nan)
    )

    g["ema9"] = (
        g.close
        .ewm(span=9, adjust=False)
        .mean()
    )

    g["ema20"] = (
        g.close
        .ewm(span=20, adjust=False)
        .mean()
    )

    first = g.iloc[0]
    last = g.iloc[-1]

    high_index = g.high.idxmax()
    low_index = g.low.idxmin()

    high = float(g.loc[high_index, "high"])
    low = float(g.loc[low_index, "low"])

    opening_price = float(first.open)
    closing_price = float(last.close)

    day_range = max(high - low, 1e-9)

    result = {

        "symbol": str(first.symbol),

        "bars": int(len(g)),

        "open": round(opening_price, 4),

        "high": round(high, 4),

        "low": round(low, 4),

        "close": round(closing_price, 4),

        "high_time_et":
            str(g.loc[high_index, "timestamp"]),

        "low_time_et":
            str(g.loc[low_index, "timestamp"]),

        "open_to_close_pct":
            round(
                (closing_price / opening_price - 1) * 100,
                3
            ),

        "mfe_long_from_open_pct":
            round(
                (high / opening_price - 1) * 100,
                3
            ),

        "mae_long_from_open_pct":
            round(
                (low / opening_price - 1) * 100,
                3
            ),

        "mfe_short_from_open_pct":
            round(
                (opening_price - low)
                / opening_price * 100,
                3
            ),

        "mae_short_from_open_pct":
            round(
                (high - opening_price)
                / opening_price * 100,
                3
            ),

        "close_location_in_range":
            round(
                (closing_price - low)
                / day_range,
                3
            ),

        "pct_bars_close_above_open":
            round(
                float(
                    (g.close > opening_price).mean()
                ),
                3
            ),

        "pct_bars_close_above_vwap":
            round(
                float(
                    (g.close > g.vwap).mean()
                ),
                3
            ),

        "closing_vwap":
            round(
                float(g.iloc[-1].vwap),
                4
            )
            if pd.notna(g.iloc[-1].vwap)
            else None,

        "closing_ema9":
            round(
                float(g.iloc[-1].ema9),
                4
            ),

        "closing_ema20":
            round(
                float(g.iloc[-1].ema20),
                4
            ),

        "pattern_auto":
            classify(g),
    }

    if prior_close and prior_close > 0:

        gap = (
            opening_price / prior_close - 1
        ) * 100

        close_to_close = (
            closing_price / prior_close - 1
        ) * 100

        result.update({

            "prior_close":
                round(prior_close, 4),

            "opening_gap_pct":
                round(gap, 3),

            "close_to_close_pct":
                round(close_to_close, 3),

            "gap_retention_ratio":
                round(
                    close_to_close / gap,
                    3
                )
                if abs(gap) > 0.05
                else None,
        })

    return result


def get_prior_closes(symbols):

    output = {}

    for i in range(0, len(symbols), 80):

        batch = symbols[i:i + 80]

        try:

            data = yf.download(
                batch,
                period="7d",
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                threads=True,
                progress=False,
            )

            for symbol in batch:

                try:

                    x = (
                        data[symbol]
                        if len(batch) > 1
                        else data
                    )

                    x = x.dropna(
                        subset=["Close"]
                    )

                    if len(x) >= 2:
                        output[symbol] = float(
                            x["Close"].iloc[-2]
                        )

                except Exception:
                    pass

        except Exception:
            pass

    return output


def main():

    paths = sorted(
        RAW.glob("*.csv.gz")
    )

    if not paths:
        raise RuntimeError(
            "No raw session file found."
        )

    path = paths[-1]

    df = pd.read_csv(
        path,
        parse_dates=["timestamp"]
    )

    symbols = sorted(
        df.symbol.unique()
    )

    prior_closes = get_prior_closes(
        symbols
    )

    rows = []

    for symbol, group in df.groupby("symbol"):

        rows.append(
            metrics_for_symbol(
                group,
                prior_closes.get(symbol)
            )
        )

    rows = sorted(
        rows,
        key=lambda x: x["symbol"]
    )

    session = path.name.split(".csv")[0]

    payload = {

        "session": session,

        "source":
            "Yahoo Finance/yfinance (unofficial)",

        "bar_interval": "1m",

        "regular_hours":
            "09:30-16:00 America/New_York",

        "symbols": rows,
    }

    (
        ROOT /
        "data" /
        "latest_session.json"
    ).write_text(
        json.dumps(
            payload,
            indent=2
        )
    )

    flat = pd.DataFrame(rows)

    flat.insert(
        0,
        "session",
        session
    )

    history_path = (
        ROOT /
        "data" /
        "history.csv"
    )

    if history_path.exists():

        old = pd.read_csv(
            history_path
        )

        old = old[
            old.session.astype(str)
            != session
        ]

        flat = pd.concat(
            [old, flat],
            ignore_index=True
        )

    flat.to_csv(
        history_path,
        index=False
    )

    print(
        f"Wrote {len(rows)} symbols "
        "to latest_session.json "
        "and history.csv"
    )


if __name__ == "__main__":
    main()
