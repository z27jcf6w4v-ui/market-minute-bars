from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf
import yaml

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

ET = ZoneInfo("America/New_York")


def load_symbols():
    cfg = yaml.safe_load((ROOT / "watchlist.yml").read_text())

    symbols = set(cfg.get("core", []))
    symbols |= set(cfg.get("extra", []))

    if cfg.get("include_sp500", True):
        try:
            tables = pd.read_html(
                "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            )

            sp500 = (
                tables[0]["Symbol"]
                .astype(str)
                .str.replace(".", "-", regex=False)
            )

            symbols |= set(sp500)

        except Exception as e:
            print("WARNING: S&P 500 list unavailable:", e)

    return sorted(symbols)


def download_batch(symbols):
    return yf.download(
        tickers=symbols,
        period="1d",
        interval="1m",
        group_by="ticker",
        auto_adjust=False,
        prepost=False,
        threads=True,
        progress=False,
    )


def main():

    now = datetime.now(ET)
    session = now.date().isoformat()

    symbols = load_symbols()

    print(f"Collecting {len(symbols)} symbols for {session}")

    good = []
    failed = []
    frames = []

    batch_size = 60

    for i in range(0, len(symbols), batch_size):

        batch = symbols[i:i + batch_size]

        try:
            data = download_batch(batch)

            for symbol in batch:

                try:

                    if len(batch) == 1:
                        d = data.copy()
                    else:
                        d = data[symbol].copy()

                    d = d.dropna(how="all")

                    if d.empty:
                        failed.append(symbol)
                        continue

                    d = d.reset_index()

                    dtcol = (
                        "Datetime"
                        if "Datetime" in d.columns
                        else d.columns[0]
                    )

                    d["timestamp"] = (
                        pd.to_datetime(d[dtcol], utc=True)
                        .dt.tz_convert(ET)
                    )

                    d["symbol"] = symbol

                    keep = [
                        "symbol",
                        "timestamp",
                        "Open",
                        "High",
                        "Low",
                        "Close",
                        "Volume",
                    ]

                    d = d[keep].rename(columns=str.lower)

                    frames.append(d)
                    good.append(symbol)

                except Exception:
                    failed.append(symbol)

        except Exception as e:

            print("Batch error:", e)
            failed.extend(batch)

        time.sleep(1)

    if not frames:
        raise RuntimeError("No minute bars downloaded.")

    out = pd.concat(frames, ignore_index=True)

    # Keep only regular U.S. trading session:
    # 9:30 AM through 3:59 PM New York time.
    local = out["timestamp"]

    minutes = local.dt.hour * 60 + local.dt.minute

    out = out[
        (minutes >= 570) &
        (minutes < 960)
    ].copy()

    out["timestamp"] = out["timestamp"].astype(str)

    path = RAW / f"{session}.csv.gz"

    out.to_csv(
        path,
        index=False,
        compression="gzip"
    )

    manifest = {
        "session": session,
        "requested": len(symbols),
        "symbols_with_data": len(set(good)),
        "rows": len(out),
        "failed_symbols": sorted(set(failed)),
        "generated_at_et": now.isoformat(),
    }

    (ROOT / "data" / "collection_status.json").write_text(
        json.dumps(manifest, indent=2)
    )

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
