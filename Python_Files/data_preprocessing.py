import pandas as pd
import numpy as np
import pytz

def remove_timezone(dt_str):
    return dt_str[:-4]  # Adjust slicing based on your data format

def parse_datetime(dt):
    try:
        return pd.to_datetime(dt)
    except ValueError:
        if 'CDT' in dt:
            dt = dt.replace(' CDT', '')
            parsed_dt = pd.to_datetime(dt, format='%m/%d/%Y %H:%M')
            central = pytz.timezone('America/Chicago')
            return parsed_dt.tz_localize(central).tz_convert(pytz.utc).tz_localize(None)
        elif 'UTC' in dt:
            dt = dt.replace(' UTC', '')
            return pd.to_datetime(dt, format='%m/%d/%Y %H:%M')
        else:
            return pd.to_datetime(dt)

def read_and_preprocess_csv(file_path, headers, skip_rows=8):
    df = pd.read_csv(file_path, names=headers, skiprows=skip_rows, index_col=False)
    df['Date_Time'] = df['Date_Time'].apply(remove_timezone)
    date_format = "%m/%d/%Y %H:%M"
    df['Date_Time'] = pd.to_datetime(df['Date_Time'], format=date_format, errors='coerce')
    df.rename(columns={'Date_Time': 'timestamp'}, inplace=True)
    return df

def load_and_merge_data():
    df1_headers = [
        "Station_ID", "Date_Time", "altimeter_set_1", "air_temp_set_1", "relative_humidity_set_1",
        "wind_speed_set_1", "wind_direction_set_1", "wind_gust_set_1", "solar_radiation_set_1",
        "precip_accum_24_hour_set_1", "precip_accum_since_local_midnight_set_1",
        "wind_chill_set_1d", "wind_cardinal_direction_set_1d", "heat_index_set_1d",
        "dew_point_temperature_set_1d", "pressure_set_1d", "sea_level_pressure_set_1d"
    ]

    df2_headers = [
        "Station_ID", "Date_Time", "altimeter_set_1", "air_temp_set_1", "dew_point_temperature_set_1",
        "relative_humidity_set_1", "wind_speed_set_1", "wind_direction_set_1", "wind_gust_set_1",
        "sea_level_pressure_set_1", "weather_cond_code_set_1", "cloud_layer_3_code_set_1",
        "pressure_tendency_set_1", "precip_accum_one_hour_set_1", "precip_accum_three_hour_set_1",
        "cloud_layer_1_code_set_1", "cloud_layer_2_code_set_1", "precip_accum_six_hour_set_1",
        "precip_accum_24_hour_set_1", "visibility_set_1", "metar_remark_set_1", "metar_set_1",
        "air_temp_high_6_hour_set_1", "air_temp_low_6_hour_set_1", "peak_wind_speed_set_1",
        "ceiling_set_1", "pressure_change_code_set_1", "air_temp_high_24_hour_set_1",
        "air_temp_low_24_hour_set_1", "peak_wind_direction_set_1", "wind_chill_set_1d",
        "wind_cardinal_direction_set_1d", "heat_index_set_1d", "weather_condition_set_1d",
        "weather_summary_set_1d", "cloud_layer_1_set_1d", "cloud_layer_2_set_1d",
        "cloud_layer_3_set_1d", "dew_point_temperature_set_1d", "pressure_set_1d",
        "sea_level_pressure_set_1d"
    ]

    df1 = read_and_preprocess_csv("data/G3425.csv", df1_headers)
    df2 = read_and_preprocess_csv("data/KHYI.csv", df2_headers)
    
    df3 = pd.read_csv("data/Meadow Center Sensor Data Test.csv")
    df3 = df3.drop(columns=['Month', 'Day', 'Year', 'Date'])
    df3.rename(columns={'Taken At': 'timestamp', 'Temperature': 'Water Temperature'}, inplace=True)
    
    df4 = pd.read_csv('data/usgs.waterservices.csv', skiprows=1)
    df4.rename(columns={'20d': 'timestamp'}, inplace=True)

    dfs = [df1, df2, df3, df4]
    for dataframe in dfs:
        dataframe['timestamp'] = dataframe['timestamp'].apply(parse_datetime)
        dataframe['timestamp'] = dataframe['timestamp'].dt.tz_localize(None)

    df1.set_index('timestamp', inplace=True)
    df2.set_index('timestamp', inplace=True)
    df3.set_index('timestamp', inplace=True)
    df4.set_index('timestamp', inplace=True)

    merged_df = pd.concat([df1, df2], axis=0)
    merged_df = pd.merge(merged_df, df3, on='timestamp', how='outer')
    merged_df = pd.merge(merged_df, df4, on='timestamp', how='outer')
    merged_df.reset_index(inplace=True)

    return merged_df

def preprocess_merged_data(merged_df):
    merged_df.drop(columns=['5s', '15s', '6s', '10s'], inplace=True)
    dividing_line = np.array([[29.89303, -97.932837], [29.89310, -97.932837]])
    merged_df['Cluster'] = np.where(merged_df['Lat'] < dividing_line[0, 0], 'Downstream', 'Upstream')
    merged_df.set_index('timestamp', inplace=True)
    merged_df = merged_df[~merged_df.index.duplicated(keep='last')]

    all_timestamps = pd.date_range(start='2022-06-11 00:00:00', end='2023-06-11 00:00:00', freq='15S')
    merged_df = merged_df.reindex(all_timestamps, fill_value=pd.NA)
    merged_df.rename(columns={'index': 'timestamp', '14n': 'Discharge Rate'}, inplace=True)

    for col_name in merged_df.columns:
        if pd.api.types.is_numeric_dtype(merged_df[col_name]):
            col_mean = merged_df[col_name].mean()
            if not np.isnan(col_mean):
                first_nan_index = merged_df[col_name].index[merged_df[col_name].isna()].min()
                last_nan_index = merged_df[col_name].index[merged_df[col_name].isna()].max()
                if first_nan_index is not np.nan:
                    merged_df.at[first_nan_index, col_name] = col_mean
                if last_nan_index is not np.nan:
                    merged_df.at[last_nan_index, col_name] = col_mean

    merged_df.interpolate(method='linear', limit_direction='forward', inplace=True)
    non_numeric_columns = merged_df.select_dtypes(exclude='number').columns
    merged_df[non_numeric_columns] = merged_df[non_numeric_columns].fillna(method='ffill')
    merged_df[non_numeric_columns] = merged_df[non_numeric_columns].fillna(method='bfill')
    merged_df.dropna(axis=1, how='any', inplace=True)

    merged_df = merged_df[(merged_df['TDS'] >= 200) & (merged_df['TDS'] <= 1000)]

    return merged_df

def sample_and_save_data(merged_df, output_file):
    sampled_df = merged_df.sample(frac=0.005)
    sampled_df.to_csv(output_file, index=False)

def main():
    merged_df = load_and_merge_data()
    preprocessed_df = preprocess_merged_data(merged_df)
    sample_and_save_data(preprocessed_df, 'data/Final_Data_Frame.csv')
    print("Data preprocessing completed. Output saved to /mnt/data/Final_Data_Frame.csv")

if __name__ == "__main__":
    main()
