import numpy as np
import pandas as pd
import ta # type: ignore
import matplotlib.pyplot as plt


### Time in UTC, donc il y a un décalage horaire ###
market_open_time = "13:30:00"
market_close_time = "20:00:00"


def load_data(data_path):
    df = pd.read_parquet(data_path)
    df = df.reset_index().set_index("ts_event") # pour avoir ts_event en index
    df.drop(columns=["publisher_id", "ts_recv", "rtype","instrument_id","date","symbol"], inplace=True) # because it's unuseful
    return df


def calculate_mid_price(df):
    df["mid_price"] = df[["ask_px_00", "bid_px_00"]].mean(axis=1)
    df["mid_price"] = df["mid_price"].combine_first(df["ask_px_00"])
    df["mid_price"] = df["mid_price"].combine_first(df["bid_px_00"])
    return df


def resample_mid_prices(df, sampling_rate):
    mid_prices = pd.DataFrame(
        {
            "mid_price_high": df["mid_price"].resample(sampling_rate).max().ffill(),
            "mid_price_low": df["mid_price"].resample(sampling_rate).min().ffill(),
            "mid_price_close": df["mid_price"].resample(sampling_rate).last().ffill(),
            "mid_price_open": df["mid_price"].resample(sampling_rate).first().ffill(),
        }
    )
    mid_prices["Returns"] = mid_prices["mid_price_close"].pct_change()
    mid_prices["Target"] = np.sign(mid_prices["Returns"])
    return mid_prices


def calculate_order_sizes(df, sampling_rate): # pour avoir les aggrégats de ce qu'il s'est passé pendant la période
    grouped = (
        df.groupby([pd.Grouper(freq=sampling_rate), "action", "side"])["size"]
        .sum()
        .reset_index()
    )
    order_sizes = grouped.pivot_table(
        index="ts_event", columns=["action", "side"], values="size", fill_value=0
    )
    columns_to_keep = [
        ("A", "A"),
        ("A", "B"),
        ("C", "A"),
        ("C", "B"),
        ("T", "A"),
        ("T", "B"),
    ]
    order_sizes = order_sizes[columns_to_keep]

    action_mapping = {"A": "add", "C": "cancel", "T": "trade"}
    side_mapping = {"A": "ask", "B": "bid"}
    order_sizes.columns = [
        f"{action_mapping[action]}_{side_mapping[side]}_size"
        for action, side in order_sizes.columns
    ]
    order_sizes["net_ask_size"] = (
        order_sizes["add_ask_size"] - order_sizes["cancel_ask_size"]
    )
    order_sizes["net_bid_size"] = (
        order_sizes["add_bid_size"] - order_sizes["cancel_bid_size"]
    )

    return order_sizes


def preprocess_data(df, sampling_rate="1s"):
    df = calculate_mid_price(df)
    mid_prices = resample_mid_prices(df, sampling_rate)
    order_sizes = calculate_order_sizes(df, sampling_rate)
    order_sizes = order_sizes.reindex(mid_prices.index, fill_value=0)

    df_combined = pd.concat([mid_prices, order_sizes], axis=1)
    df_combined.dropna(inplace=True)

    return df_combined


def compute_hft_indicators(df):
    indicators = df.copy()

    indicators["EMA_5"] = ta.trend.ema_indicator(
        indicators["mid_price_close"], window=5
    )
    indicators["MA_5"] = (
        indicators["mid_price_close"].rolling(window=5, min_periods=1).mean()
    )

    indicators["Bollinger_Upper"] = indicators["MA_5"] + (
        indicators["mid_price_close"].rolling(5).std() * 2
    )
    indicators["Bollinger_Lower"] = indicators["MA_5"] - (
        indicators["mid_price_close"].rolling(5).std() * 2
    )

    indicators["High_Shift"] = indicators["mid_price_high"].shift(1)
    indicators["Low_Shift"] = indicators["mid_price_low"].shift(1)

    indicators["DMP_3"] = (
        pd.Series(
            np.where(
                (
                    indicators["mid_price_high"] - indicators["High_Shift"]
                    > indicators["Low_Shift"] - indicators["mid_price_low"]
                ),
                np.maximum(indicators["mid_price_high"] - indicators["High_Shift"], 0),
                0,
            ),
            index=df.index,
        )
        .rolling(3, min_periods=1)
        .sum()
    )

    indicators["DMN_3"] = (
        pd.Series(
            np.where(
                (
                    indicators["Low_Shift"] - indicators["mid_price_low"]
                    > indicators["mid_price_high"] - indicators["High_Shift"]
                ),
                np.maximum(indicators["Low_Shift"] - indicators["mid_price_low"], 0),
                0,
            ),
            index=df.index,
        )
        .rolling(3, min_periods=1)
        .sum()
    )

    indicators["OLL3"] = (
        indicators["mid_price_open"]
        - indicators["mid_price_low"].rolling(3, min_periods=1).min()
    )
    indicators["OLL5"] = (
        indicators["mid_price_open"]
        - indicators["mid_price_low"].rolling(5, min_periods=1).min()
    )

    indicators["STOCHk_7_3_3"] = ta.momentum.stoch(
        indicators["mid_price_high"],
        indicators["mid_price_low"],
        indicators["mid_price_close"],
        window=7,
        smooth_window=3,
    )
    indicators["STOCHd_7_3_3"] = (
        indicators["STOCHk_7_3_3"].rolling(3, min_periods=1).mean()
    )

    indicators.drop(columns=["High_Shift", "Low_Shift"], inplace=True)

    indicators.ffill(inplace=True)

    last_nan_index = indicators[indicators.isna().any(axis=1)].index[-1]

    # Drop all starting values with NaNs
    indicators = indicators.iloc[indicators.index.get_loc(last_nan_index) + 1 :]

    return indicators.between_time(market_open_time, market_close_time)


def combine_data(data_paths, sampling_rate="1s"):
    trading_days_df = []

    for file_path in data_paths:
        df = load_data(file_path)
        df = preprocess_data(df, sampling_rate)
        df_hft = compute_hft_indicators(df)

        trading_days_df.append(df_hft)

    return pd.concat(trading_days_df, axis=0)


def add_time_features(combined_df):
    combined_df = combined_df.copy()

    # Compute market open time (09:30 AM) for each trading day
    combined_df["market_open_time"] = combined_df.index.normalize() + pd.Timedelta(
        hours=9, minutes=30
    )

    # Compute seconds since market open
    combined_df["time_since_open"] = (
        combined_df.index - combined_df["market_open_time"]
    ).dt.total_seconds()

    # Encode day of the week as one-hot vectors
    combined_df["day_of_week"] = combined_df.index.weekday  # Extract day of the week
    combined_df = pd.get_dummies(combined_df, columns=["day_of_week"], prefix="dow")

    one_hot_columns = [col for col in combined_df.columns if col.startswith("dow_")]
    combined_df[one_hot_columns] = combined_df[one_hot_columns].astype(int)

    # Add market session feature (morning = 0, afternoon = 1)
    combined_df["market_session"] = (
        combined_df["time_since_open"] > 3.5 * 3600
    ).astype(int)

    # Drop unnecessary columns
    combined_df.drop(columns=["market_open_time"], inplace=True)

    return combined_df
