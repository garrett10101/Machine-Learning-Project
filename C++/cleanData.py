# Let's redo the data cleaning script properly and execute it step by step.
import pandas as pd
import numpy as np

# Load the provided dataset
df = pd.read_csv('C++/output_data.csv')
#Drop Station_ID
df = df.drop('Station_ID', axis=1)
# Drop columns with more than 75% NaN values
threshold = len(df) * 0.75
df = df.dropna(thresh=threshold, axis=1)

# Separate numeric and non-numeric columns explicitly
numeric_cols = df.select_dtypes(include=[np.number]).columns
non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns

# Handle numeric columns: interpolate missing values
df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')

# Handle non-numeric columns: fill NaN with forward-fill then backward-fill, then encode
df[non_numeric_cols] = df[non_numeric_cols].fillna(method='ffill').fillna(method='bfill')

# Encode non-numeric columns numerically
for col in non_numeric_cols:
    df[col], _ = pd.factorize(df[col])

# Check for any remaining NaN values and drop rows with NaNs if necessary (max 10% drop allowed)
remaining_nans = df.isnull().sum().sum()
rows_before = len(df)
if remaining_nans > 0:
    df.dropna(inplace=True)
    rows_after = len(df)
    dropped_percentage = (rows_before - rows_after) / rows_before * 100
    if dropped_percentage > 10:
        print(f"Warning: {dropped_percentage:.2f}% of data would be dropped, exceeding your 10% threshold.")
    else:
        print(f"{dropped_percentage:.2f}% of data was dropped due to remaining NaN values.")
else:
    print("No NaN values remain after cleaning.")

# Save the cleaned dataset
cleaned_csv_path = 'C++/cleaned_output_data.csv'
df.to_csv(cleaned_csv_path, index=False)

# Display basic info of cleaned data
df.info(), cleaned_csv_path
