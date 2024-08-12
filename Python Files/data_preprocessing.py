import pandas as pd
import numpy as np
import pytz

def preprocess_data(input_files, output_file):
    # Load input files
    df1 = pd.read_csv(input_files['df1'], names=[
        "Station_ID", "Date_Time", "altimeter_set_1", "air_temp_set_1", "relative_humidity_set_1",
        "wind_speed_set_1", "wind_direction_set_1", "wind_gust_set_1", "solar_radiation_set_1",
        "precip_accum_24_hour_set_1", "precip_accum_since_local_midnight_set_1",
        "wind_chill_set_1d", "wind_cardinal_direction_set_1d", "heat_index_set_1d",
        "dew_point_temperature_set_1d", "pressure_set_1d", "sea_level_pressure_set_1d"
    ], skiprows=8, index_col=False)

    df2 = pd.read_csv(input_files['df2'], names=[
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
    ], skiprows=8, index_col=False)

    df3 = pd.read_csv(input_files['df3'])
    df4 = pd.read_csv(input_files['df4'], skiprows=1)

    # Remove timezone from Date_Time
    def remove_timezone(dt_str):
        return dt_str[:-4]

    df1['Date_Time'] = df1['Date_Time'].apply(remove_timezone)
    df2['Date_Time'] = df2['Date_Time'].apply(remove_timezone)

    date_format = "%m/%d/%Y %H:%M"

    df1['Date_Time'] = pd.to_datetime(df1['Date_Time'], format=date_format, errors='coerce')
    df2['Date_Time'] = pd.to_datetime(df2['Date_Time'], format=date_format, errors='coerce')

    df1.rename(columns={'Date_Time': 'timestamp'}, inplace=True)
    df2.rename(columns={'Date_Time': 'timestamp'}, inplace=True)

    df3 = df3.drop(columns=['Month', 'Day', 'Year', 'Date'])
    df3.rename(columns={'Taken At': 'timestamp'}, inplace=True)

    df4.rename(columns={'20d': 'timestamp'}, inplace=True)

    dfs = [df1, df2, df3, df4]

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

    for dataframe in dfs:
        dataframe['timestamp'] = dataframe['timestamp'].apply(parse_datetime)

    df1['timestamp'] = df1['timestamp'].dt.tz_localize(None)
    df2['timestamp'] = df2['timestamp'].dt.tz_localize(None)
    df3['timestamp'] = df3['timestamp'].dt.tz_localize(None)
    df4['timestamp'] = df4['timestamp'].dt.tz_localize(None)

    df1.set_index('timestamp', inplace=True)
    df2.set_index('timestamp', inplace=True)
    df3.set_index('timestamp', inplace=True)
    df4.set_index('timestamp', inplace=True)

    merged_df = pd.concat([df1, df2], axis=0)
    merged_df = pd.merge(merged_df, df3, on='timestamp', how='outer')
    merged_df = pd.merge(merged_df, df4, on='timestamp', how='outer')
    merged_df.reset_index(inplace=True)

    final_df = merged_df.drop(columns=['5s', '15s', '6s', '10s'])

    dividing_line = np.array([[29.89303, -97.932837], [29.89310, -97.932837]])
    final_df['Cluster'] = np.where(final_df['Lat'] < dividing_line[0, 0], 'Downstream', 'Upstream')

    final_df.set_index('timestamp', inplace=True)
    final_df = final_df[~final_df.index.duplicated(keep='last')]
    all_timestamps = pd.date_range(start='2022-06-11 00:00:00', end='2023-06-11 00:00:00', freq='15s')
    final_df = final_df.reindex(all_timestamps, fill_value=pd.NA)
    final_df = final_df.rename(columns={'14n': 'Discharge Rate'})

    final_df = final_df.infer_objects()

    for col_name in final_df.columns:
        if pd.api.types.is_numeric_dtype(final_df[col_name]):
            col_mean = final_df[col_name].mean()
            if not np.isnan(col_mean):
                first_nan_index = final_df[col_name].index[final_df[col_name].isna()].min()
                last_nan_index = final_df[col_name].index[final_df[col_name].isna()].max()
                if first_nan_index is not np.nan:
                    final_df.at[first_nan_index, col_name] = col_mean
                if last_nan_index is not np.nan:
                    final_df.at[last_nan_index, col_name] = col_mean

    final_df.interpolate(method='linear', limit_direction='forward', inplace=True)

    non_numeric_columns = final_df.select_dtypes(exclude='number').columns
    final_df[non_numeric_columns] = final_df[non_numeric_columns].ffill()
    final_df[non_numeric_columns] = final_df[non_numeric_columns].bfill()

    final_df = final_df.dropna(axis=1, how='any')
    final_df = final_df[(final_df['TDS'] >= 200) & (final_df['TDS'] <= 1000)]

    final_df.to_csv(output_file, index=False)

files = {'df1':"Data/G3425.csv", 'df2':"Data/KHYI.csv", 'df3':"Data/Meadow Center Sensor Data Test.csv", 'df4':"Data/usgs.waterservices.csv"}
preprocess_data(files, 'Data/output_data.csv')