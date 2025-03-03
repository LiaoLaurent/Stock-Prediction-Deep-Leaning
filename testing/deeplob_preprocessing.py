import pandas as pd
import numpy as np
import os 

market_open = pd.Timestamp("09:30:00").time()
market_close = pd.Timestamp("16:00:00").time()

def load_data(data_path):
    df = pd.read_parquet(data_path)
    return df.between_time(market_open, market_close)

def resample_lob(raw_data, sampling_rate):

    df = raw_data.copy()

    df["mid_price"] = df[["bid_px_00", "ask_px_00"]].mean(axis=1)
    df["mid_price"] = df["mid_price"].combine_first(df["bid_px_00"])
    df["mid_price"] = df["mid_price"].combine_first(df["ask_px_00"])
    
    lob_columns = [col for col in df.columns if col.startswith("bid_px_") or 
                   col.startswith("ask_px_") or col.startswith("bid_sz_") or 
                   col.startswith("ask_sz_")]
    df_lob = df[lob_columns].resample(sampling_rate).last()

    mid_prices_last = df["mid_price"].resample(sampling_rate).last()

    mid_price_variation = mid_prices_last.pct_change(fill_method=None)
    mid_price_variation_class = np.sign(mid_price_variation) + 1
    mid_price_variation_class.name = "mid_price_variation_class"

    df_resampled = pd.concat(
        [df_lob, mid_price_variation_class], axis=1
    ).dropna()

    return df_resampled

def process_and_combine_data(start_date, end_date, data_folder="../AAPL_data/DP_MBP_10/AAPL", sampling_rate="1s"):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    date_range = pd.bdate_range(start=start_date, end=end_date)
    
    data_paths = [
        f"{data_folder}/AAPL_{date.strftime('%Y-%m-%d')}_xnas-itch.parquet"
        for date in date_range
    ]
    existing_data_paths = [path for path in data_paths if os.path.exists(path)]
    processed_data = []
    
    for path in existing_data_paths:
        try:
            df = load_data(path)
            df_resampled = resample_lob(df, sampling_rate=sampling_rate)
            processed_data.append(df_resampled)
        except Exception as e:
            print(f"Error with the file {path}: {e}")
    
    if processed_data:
        all_data = pd.concat(processed_data)
    else:
        all_data = pd.DataFrame()
    
    return all_data