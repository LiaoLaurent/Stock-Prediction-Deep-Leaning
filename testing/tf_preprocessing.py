import pandas as pd
import ta
import numpy as np

market_open = pd.Timestamp("09:30:00").time()
market_close = pd.Timestamp("16:00:00").time()


def load_data(data_path):
    df = pd.read_parquet(data_path)
    df = df.reset_index().set_index("ts_event")
    return df.between_time(market_open, market_close)


def resample_mid_prices(df, sampling_rate):
    """
    Resample the mid_price values by the given sampling rate to get highest, lowest, open, close, and mean prices.

    Parameters:
    df (pd.DataFrame): DataFrame containing the mid_price and depth columns.
    sampling_rate (str): The sampling rate for resampling the data.

    Returns:
    pd.DataFrame: DataFrame containing the resampled mid_price values.
    """
    df_copy = df.copy()
    # Add mid price column (either the bid or the ask if one is missing)
    df_copy["mid_price"] = df_copy[["ask_px_00", "bid_px_00"]].mean(axis=1)
    df_copy["mid_price"] = df_copy["mid_price"].combine_first(df_copy["ask_px_00"])
    df_copy["mid_price"] = df_copy["mid_price"].combine_first(df_copy["bid_px_00"])

    mid_prices_high = df_copy["mid_price"].resample(sampling_rate).max().ffill()
    mid_prices_low = df_copy["mid_price"].resample(sampling_rate).min().ffill()
    mid_prices_close = df_copy["mid_price"].resample(sampling_rate).last().ffill()
    mid_prices_open = df_copy["mid_price"].resample(sampling_rate).first().ffill()

    returns_volatilities = (
        df_copy["mid_price"].pct_change().resample(sampling_rate).std().ffill()
    )

    returns = mid_prices_close.pct_change()

    # Combine the resampled mid_price values into a single DataFrame
    mid_prices = pd.DataFrame(
        {
            "mid_price_high": mid_prices_high,
            "mid_price_low": mid_prices_low,
            "mid_price_close": mid_prices_close,
            "mid_price_open": mid_prices_open,
            "returns": returns,
            "returns_volatility": returns_volatilities,
        }
    )

    return mid_prices


def group_and_pivot_order_sizes(df, sampling_rate):
    """
    Group by ts_event, action, and side, then sum the sizes and pivot the table to create new columns for each combination of action and side.

    Parameters:
    df (pd.DataFrame): DataFrame containing the ts_event, action, side, and size columns.
    sampling_rate (str): The sampling rate for grouping the data.

    Returns:
    pd.DataFrame: DataFrame with pivoted columns for each combination of action and side.
    """
    # Group by ts_event, action, and side, then sum the sizes
    grouped = (
        df.groupby([pd.Grouper(freq=sampling_rate), "action", "side"])["size"]
        .sum()
        .reset_index()
    )

    # Pivot the table to create new columns for each combination of action and side
    order_sizes = grouped.pivot_table(
        index="ts_event", columns=["action", "side"], values="size", fill_value=0
    )

    # Drop unnecessary columns
    columns_to_keep = [
        ("A", "A"),
        ("A", "B"),
        ("C", "A"),
        ("C", "B"),
        ("T", "A"),
        ("T", "B"),
    ]
    order_sizes = order_sizes[columns_to_keep]

    # Define mappings for action and side
    action_mapping = {"A": "add", "C": "cancel", "T": "trade"}
    side_mapping = {"A": "ask", "B": "bid"}

    # Rename columns
    order_sizes.columns = [
        f"{action_mapping[action]}_{side_mapping[side]}_size"
        for action, side in order_sizes.columns
    ]

    order_sizes["net_add_ask_size"] = (
        order_sizes["add_ask_size"] - order_sizes["cancel_ask_size"]
    )
    order_sizes["net_add_bid_size"] = (
        order_sizes["add_bid_size"] - order_sizes["cancel_bid_size"]
    )

    order_sizes.drop(
        columns=["add_ask_size", "add_bid_size", "cancel_ask_size", "cancel_bid_size"],
        inplace=True,
    )

    return order_sizes


def compute_technical_indicators(df):
    """
    Compute technical indicators for a given DataFrame with OHLC-like structure.
    """
    # Make a copy of the DataFrame to avoid modifying the original
    df = df.copy()

    df["Returns"] = df["mid_price_close"].pct_change()
    df["Target_close"] = np.sign(df["Returns"]) + 1

    ### TREND INDICATORS ###
    df["ADX_5"] = ta.trend.ADXIndicator(
        df["mid_price_high"], df["mid_price_low"], df["mid_price_close"], window=5
    ).adx()
    df["ADX_7"] = ta.trend.ADXIndicator(
        df["mid_price_high"], df["mid_price_low"], df["mid_price_close"], window=7
    ).adx()
    df["ADX_10"] = ta.trend.ADXIndicator(
        df["mid_price_high"], df["mid_price_low"], df["mid_price_close"], window=10
    ).adx()
    df["DMP_5"] = ta.trend.ADXIndicator(
        df["mid_price_high"], df["mid_price_low"], df["mid_price_close"], window=5
    ).adx_pos()
    df["DMP_10"] = ta.trend.ADXIndicator(
        df["mid_price_high"], df["mid_price_low"], df["mid_price_close"], window=10
    ).adx_pos()
    df["DMN_5"] = ta.trend.ADXIndicator(
        df["mid_price_high"], df["mid_price_low"], df["mid_price_close"], window=5
    ).adx_neg()
    df["DMN_10"] = ta.trend.ADXIndicator(
        df["mid_price_high"], df["mid_price_low"], df["mid_price_close"], window=10
    ).adx_neg()

    # Aroon Indicator requires both high and low prices
    df["AROONU_7"] = ta.trend.AroonIndicator(
        df["mid_price_high"], df["mid_price_low"], window=7
    ).aroon_up()
    df["AROOND_7"] = ta.trend.AroonIndicator(
        df["mid_price_high"], df["mid_price_low"], window=7
    ).aroon_down()

    # Open-Low and Open-High calculations
    df["OLL3"] = df["mid_price_open"] - df["mid_price_low"].rolling(window=3).min()
    df["OLL5"] = df["mid_price_open"] - df["mid_price_low"].rolling(window=5).min()
    df["OLL10"] = df["mid_price_open"] - df["mid_price_low"].rolling(window=10).min()
    df["OLL15"] = df["mid_price_open"] - df["mid_price_low"].rolling(window=15).min()
    df["OHH3"] = df["mid_price_high"].rolling(window=3).max() - df["mid_price_open"]
    df["OHH5"] = df["mid_price_high"].rolling(window=5).max() - df["mid_price_open"]

    ### OSCILLATORS ###
    df["STOCHk_7_3_3"] = ta.momentum.StochasticOscillator(
        df["mid_price_high"],
        df["mid_price_low"],
        df["mid_price_close"],
        window=7,
        smooth_window=3,
    ).stoch()
    df["STOCHd_7_3_3"] = ta.momentum.StochasticOscillator(
        df["mid_price_high"],
        df["mid_price_low"],
        df["mid_price_close"],
        window=7,
        smooth_window=3,
    ).stoch_signal()

    # Avoid NaN Stochastic values
    df["STOCHk_7_3_3"] = df["STOCHk_7_3_3"].ffill()
    df["STOCHd_7_3_3"] = df["STOCHd_7_3_3"].ffill()

    df["MACD_8_21_5"] = ta.trend.MACD(
        df["mid_price_close"], window_slow=21, window_fast=8, window_sign=5
    ).macd_diff()
    df["RSI_7"] = ta.momentum.RSIIndicator(df["mid_price_close"], window=7).rsi()
    df["AO_5_10"] = ta.momentum.AwesomeOscillatorIndicator(
        df["mid_price_high"], df["mid_price_low"], window1=5, window2=10
    ).awesome_oscillator()

    ### MOVING AVERAGES ###
    df["EMA_15"] = ta.trend.EMAIndicator(
        df["mid_price_close"], window=15
    ).ema_indicator()
    df["HMA_10"] = ta.trend.WMAIndicator(
        df["mid_price_close"], window=10
    ).wma()  # HMA is not directly available in 'ta', using WMA as a placeholder
    df["KAMA_3_2_10"] = ta.momentum.KAMAIndicator(
        df["mid_price_close"], window=3, pow1=2, pow2=10
    ).kama()
    df["MA_10"] = ta.trend.SMAIndicator(
        df["mid_price_close"], window=10
    ).sma_indicator()
    df["MA_20"] = ta.trend.SMAIndicator(
        df["mid_price_close"], window=20
    ).sma_indicator()

    # Rolling CO (Close - Open)
    for w in [3, 4, 5, 6]:
        df[f"rmCO({w})"] = (
            (df["mid_price_close"] - df["mid_price_open"]).rolling(window=w).mean()
        )

    ### VOLATILITY INDICATORS ###
    df["Bollinger_Upper"] = ta.volatility.BollingerBands(
        df["mid_price_close"], window=20, window_dev=2
    ).bollinger_hband()
    df["Bollinger_Lower"] = ta.volatility.BollingerBands(
        df["mid_price_close"], window=20, window_dev=2
    ).bollinger_lband()
    df["U_minus_L"] = df["Bollinger_Upper"] - df["Bollinger_Lower"]
    df["MA20dSTD"] = df["mid_price_close"].rolling(window=20).std()

    ### OTHER INDICATORS ###
    df["CO"] = df["mid_price_close"] - df["mid_price_open"]
    df["C1O1"] = df["CO"].shift(1)
    df["C2O2"] = df["CO"].shift(2)
    df["C3O3"] = df["CO"].shift(3)
    df["range"] = df["mid_price_high"] - df["mid_price_low"]
    df["OH1"] = df["mid_price_high"].shift(1) - df["mid_price_open"].shift(1)

    return df.dropna()


def add_time_features(combined_df):
    """
    Add time-based features to the combined DataFrame.

    Parameters:
        combined_df (pd.DataFrame): The input DataFrame with a DateTime index.

    Returns:
        combined_df (pd.DataFrame): The DataFrame with added time-based features.
    """
    combined_df = combined_df.copy()

    # Add market open time for each day (9:30 AM)
    combined_df["market_open_time"] = combined_df.index.normalize() + pd.Timedelta(
        hours=9, minutes=30
    )

    # Compute seconds since market open
    combined_df["time_since_open"] = (
        combined_df.index - combined_df["market_open_time"]
    ).dt.total_seconds()

    # Encode day of the week as one-hot vectors
    combined_df["day_of_week"] = (
        combined_df.index.weekday
    )  # Extract day of the week (0=Monday, 6=Sunday)
    combined_df = pd.get_dummies(combined_df, columns=["day_of_week"], prefix="dow")

    # Convert one-hot columns to integers (0 or 1)
    one_hot_columns = [col for col in combined_df.columns if col.startswith("dow_")]
    combined_df[one_hot_columns] = combined_df[one_hot_columns].astype(int)

    # Add market session feature (morning = 0, afternoon = 1)
    combined_df["market_session"] = (
        combined_df["time_since_open"] > 3.5 * 3600  # 3.5 hours after market open
    ).astype(int)

    combined_df.drop(columns=["market_open_time"], inplace=True)

    return combined_df


def process_and_combine_data(
    start_date,
    end_date,
    data_folder="/home/janis/3A/EA/HFT_QR_RL/data/smash4/DB_MBP_10/AAL",
    sampling_rate="1s",
):
    """
    Process and combine trade data for a given date range.

    Parameters:
        start_date (str or datetime): Start date of the date range (inclusive).
        end_date (str or datetime): End date of the date range (inclusive).
        market_open (str): Market open time (e.g., "09:30").
        market_close (str): Market close time (e.g., "16:00").
        data_folder (str): Folder containing the parquet files.
        sampling_rate (str): Resampling rate for mid prices and order sizes (e.g., "5s").

    Returns:
        all_data (pd.DataFrame): Combined DataFrame containing processed data for all trade days.
    """
    # Convert start_date and end_date to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Generate business date range
    date_range = pd.bdate_range(start=start_date, end=end_date)

    # Generate file paths
    data_paths = [
        f"{data_folder}/AAL_{date.strftime('%Y-%m-%d')}.parquet"
        for date in date_range
    ]

    # Initialize list to store daily data
    trade_days_data = []

    # Process each file
    for path in data_paths:
        # Load data
        df = load_data(path)

        # Compute order sizes
        order_sizes = group_and_pivot_order_sizes(df, sampling_rate=sampling_rate)

        # Compute mid prices
        mid_prices = resample_mid_prices(df, sampling_rate=sampling_rate)

        # Reindex order sizes to match mid prices
        order_sizes = order_sizes.reindex(mid_prices.index, fill_value=0)

        # Compute technical indicators
        technical_indicators = compute_technical_indicators(mid_prices)

        df_combined = order_sizes.join(technical_indicators, how="inner")

        trade_days_data.append(df_combined)

    all_data = pd.concat(trade_days_data)

    return add_time_features(all_data)
