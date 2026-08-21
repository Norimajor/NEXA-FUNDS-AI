from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data")

TIMEFRAMES = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
}


# Start with the 7 major pairs.
PAIRS = [
    "EURUSD.m",
    "GBPUSD.m",
    "USDJPY.m",
    "USDCHF.m",
    "USDCAD.m",
    "AUDUSD.m",
    "NZDUSD.m",
]


# MT5 is configured for a very large history.
# We retrieve data in safe chunks.
CHUNK_SIZE = 5000

# Maximum chunks per timeframe.
# 1000 × 5000 = 5,000,000 candles.
#
# The downloader stops automatically when MT5
# returns fewer than CHUNK_SIZE candles.
MAX_CHUNKS = 1000


# ============================================================
# MT5 INITIALIZATION
# ============================================================

def initialize_mt5():

    if not mt5.initialize():

        raise RuntimeError(
            f"MT5 initialization failed: {mt5.last_error()}"
        )

    terminal = mt5.terminal_info()

    print()
    print("=" * 60)
    print("MT5 CONNECTED")
    print("=" * 60)

    print("Terminal:", terminal.name)
    print("Build:", terminal.build)
    print("Path:", terminal.path)
    print("Max bars:", terminal.maxbars)


# ============================================================
# SYMBOL
# ============================================================

def symbol_exists(symbol):

    return mt5.symbol_info(symbol) is not None


# ============================================================
# OUTPUT FILE
# ============================================================

def get_output_file(symbol, timeframe_name):

    filename = (
        f"{symbol.replace('.', '_')}"
        f"_{timeframe_name}.csv"
    )

    return DATA_DIR / filename


# ============================================================
# DOWNLOAD ONE TIMEFRAME
# ============================================================

def download_timeframe(
    symbol,
    timeframe_name,
    timeframe,
):

    output_file = get_output_file(
        symbol,
        timeframe_name,
    )

    print()
    print("=" * 60)
    print(
        f"[{symbol}] {timeframe_name}"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # SYMBOL CHECK
    # --------------------------------------------------------

    if not symbol_exists(symbol):

        print(
            f"[SKIP] Symbol not found: {symbol}"
        )

        return None

    if not mt5.symbol_select(
        symbol,
        True,
    ):

        print(
            f"[ERROR] Could not select {symbol}"
        )

        return None

    # --------------------------------------------------------
    # TEMP FILE
    #
    # We write to .tmp first.
    # The real CSV is only replaced after
    # the complete download succeeds.
    # --------------------------------------------------------

    temp_file = output_file.with_suffix(
        ".csv.tmp"
    )

    if temp_file.exists():

        try:
            temp_file.unlink()
        except OSError:
            pass

    chunks = []

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    for chunk_number in range(MAX_CHUNKS):

        start_pos = (
            chunk_number
            * CHUNK_SIZE
        )

        print(
            f"  chunk "
            f"{chunk_number + 1}/{MAX_CHUNKS} "
            f"(position {start_pos})",
            end="",
        )

        rates = mt5.copy_rates_from_pos(
            symbol,
            timeframe,
            start_pos,
            CHUNK_SIZE,
        )

        if rates is None:

            print()

            print(
                f"  [ERROR] "
                f"{mt5.last_error()}"
            )

            return None

        received = len(rates)

        print(
            f" -> received {received:,} candles"
        )

        if received == 0:

            print(
                "  [END] No more data."
            )

            break

        chunks.append(
            pd.DataFrame(rates)
        )

        if received < CHUNK_SIZE:

            print(
                "  [END] Reached available history."
            )

            break

    # --------------------------------------------------------
    # CHECK
    # --------------------------------------------------------

    if not chunks:

        print(
            f"[FAILED] No data for "
            f"{symbol} {timeframe_name}"
        )

        return None

    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    df = pd.concat(
        chunks,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # CLEAN
    # --------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["time"],
        unit="s",
    )

    if "tick_volume" in df.columns:

        df = df.rename(
            columns={
                "tick_volume": "volume"
            }
        )

    columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "spread",
    ]

    available_columns = [
        column
        for column in columns
        if column in df.columns
    ]

    df = df[available_columns]

    df = df.sort_values(
        "timestamp"
    )

    df = df.drop_duplicates(
        subset=["timestamp"],
        keep="last",
    )

    df = df.reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    if df.empty:

        print(
            "[FAILED] Dataframe is empty."
        )

        return None

    # --------------------------------------------------------
    # SAVE
    #
    # Write temporary file first.
    # --------------------------------------------------------

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        f"  Saving {len(df):,} candles..."
    )

    try:

        df.to_csv(
            temp_file,
            index=False,
        )

    except OSError as error:

        print()
        print(
            f"[DISK ERROR] {error}"
        )

        print(
            "Free disk space may be insufficient."
        )

        try:
            if temp_file.exists():
                temp_file.unlink()
        except OSError:
            pass

        return None

    # --------------------------------------------------------
    # ATOMIC REPLACE
    # --------------------------------------------------------

    try:

        temp_file.replace(
            output_file
        )

    except OSError as error:

        print(
            f"[ERROR] Could not finalize file: {error}"
        )

        return None

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    file_size_gb = (
        output_file.stat().st_size
        / 1_000_000_000
    )

    print()
    print(
        f"  [OK] {len(df):,} candles"
    )

    print(
        f"  Start: {df['timestamp'].iloc[0]}"
    )

    print(
        f"  End:   {df['timestamp'].iloc[-1]}"
    )

    print(
        f"  Size:  {file_size_gb:.3f} GB"
    )

    print(
        f"  Saved: {output_file}"
    )

    return df


# ============================================================
# DOWNLOAD ONE PAIR
# ============================================================

def download_pair(symbol):

    print()
    print("#" * 60)
    print(
        f"DOWNLOADING {symbol}"
    )
    print("#" * 60)

    if not symbol_exists(symbol):

        print(
            f"[SKIP] {symbol} not found."
        )

        return False

    success_count = 0

    for timeframe_name, timeframe in TIMEFRAMES.items():

        df = download_timeframe(
            symbol,
            timeframe_name,
            timeframe,
        )

        if (
            df is not None
            and not df.empty
        ):

            success_count += 1

    print()
    print(
        f"{symbol}: "
        f"{success_count}/{len(TIMEFRAMES)} "
        f"timeframes downloaded."
    )

    return success_count > 0


# ============================================================
# STORAGE REPORT
# ============================================================

def storage_report():

    print()
    print("=" * 60)
    print("DATA STORAGE")
    print("=" * 60)

    files = list(
        DATA_DIR.glob("*.csv")
    )

    if not files:

        print("No CSV files.")

        return

    total_bytes = 0

    for file in sorted(files):

        size = file.stat().st_size

        total_bytes += size

        print(
            f"{file.name:<30} "
            f"{size / 1_000_000_000:.3f} GB"
        )

    print(
        "-" * 60
    )

    print(
        f"TOTAL: "
        f"{total_bytes / 1_000_000_000:.3f} GB"
    )


# ============================================================
# DOWNLOAD EVERYTHING
# ============================================================

def download_all():

    initialize_mt5()

    results = {}

    try:

        for symbol in PAIRS:

            results[symbol] = (
                download_pair(symbol)
            )

            storage_report()

    finally:

        mt5.shutdown()

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)

    for symbol, success in results.items():

        status = (
            "OK"
            if success
            else "FAILED"
        )

        print(
            f"{symbol:<15} {status}"
        )

    storage_report()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    download_all()